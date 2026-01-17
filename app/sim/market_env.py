"""
gymnasium environment for market making RL training
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
import logging

from app.sim.market_simulator import MarketSimulator
from app.sim.orders import LimitOrder, OrderSide
from app.sim.exceptions import SimulationError

logger = logging.getLogger(__name__)


class MarketMakingEnv(gym.Env):
    """
    gymnasium environment for market making RL training

    Action space:
    multi discrete for independent bid / ask control
    bid_price_offset, bid_quantity, ask_price_offset, ask_quantity

    Observation space:
    dict space with order book and portfolio features
    flattened to 1D array for neural network input
    """

    metadata = {'render_modes': ['human'], 'render_fps': 4}

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        initial_cash: float =10000.0,
        max_steps: int = 1000,
        price_tick_size: float = 0.01,
        quantity_precision: int = 4,
        max_quantity: float = 1.0,
        n_price_levels: int = 21,
        n_quantity_levels: int = 10,
        lookback_window: int = 10
    ):
        super().__init__()

        self.symbol = symbol
        self.initial_cash = initial_cash
        self.max_steps = max_steps
        self.price_tick_size = price_tick_size
        self.quantity_precision = quantity_precision
        self.max_quantity = max_quantity
        self.n_price_levels = n_price_levels
        self.n_quantity_levels = n_quantity_levels
        self.lookback_window = lookback_window

        #initialize simulator, will load data in reset
        self.simulator: Optional[MarketSimulator] = None

        #episode tracking
        self.current_step = 0
        self.price_history: List[float] = []
        
        #track order ages for reward shaping (steps since order was placed)
        self.order_ages: Dict[str, int] = {}

        #define action space
        self.action_space = spaces.MultiDiscrete([
            n_price_levels, #bid price: -10 to +10 price ticks
            n_quantity_levels, #bid quantity: 0 to max quantity
            n_price_levels, #ask price: -10 to +10 price ticks
            n_quantity_levels #ask quantity: 0 to max quantity
        ])

        #define observation space
        #will flatten to 1D array for neural network input
        observation_dim = self._get_observation_dim()
        self.observation_space = spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (observation_dim,),
            dtype = np.float32
        )


    def _get_observation_dim(self) -> int:
        """
        calculates observation dimension
        """
        # order book features: best_bid, best_ask, mid, spread, spread_pct, imbalance = 6
        # depth: 10 levels * 2 (bid/ask) * 2 (price/qty) = 40
        # portfolio: cash, position, unrealized_pnl, return_pct = 4
        # market features: volatility, price_change = 2
        # total: 6 + 40 + 4 + 2 = 52
        return 52

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        reset the environment to the start of a new episode
        options can include: 
        start_time specific time to start
        , end_time specific time to end
        , random_start, start at a random time within the dataset
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.price_history = []
        self.order_ages = {}  #reset order ages on episode reset

        #initialize simulator
        if self.simulator is None:
            self.simulator = MarketSimulator(symbol=self.symbol, initial_cash=self.initial_cash)

            #load data use options or default
            start_time = options.get('start_time') if options else None
            end_time = options.get('end_time') if options else None

            if start_time is None:
                #default, last 7 days for better diversity
                end_time = datetime.now()
                start_time = end_time - timedelta(days=7)

            
            #load data, this is async
           
            import asyncio
            max_snapshots = options.get('max_snapshots') if options else 100000  # Increased from 50k
            sample_interval = options.get('sample_interval') if options else None
            
           
            days_diff = (end_time - start_time).days if end_time and start_time else 0
            if sample_interval is None and days_diff >= 7:
               
                sample_interval = 5 if days_diff < 14 else 10
                logger.info(f"Auto-sampling: using every {sample_interval}th snapshot to reduce data size (covering {days_diff} days)")
            
            asyncio.run(self.simulator.load_historical_data(
                start_time=start_time, 
                end_time=end_time,
                max_snapshots=max_snapshots,
                sample_interval=sample_interval
            ))
        
        start_index = options.get('start_index') if options else 0
        if options and options.get('random_start', False):
            #random start for diverse training
            max_index = len(self.simulator.snapshots) - self.max_steps
            start_index = self.np_random.integers(0, max(1,max_index))
        
        self.simulator.reset(start_index=start_index)

        #get initial observation
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        execute one step in the environment

        action interpretation::
        - action[0]: bid_price_offset (-10 to +10 ticks from mid)
        - action[1]: bid_quantity (0 to max quantity)
        - action[2]: ask_price_offset (-10 to +10 ticks from mid)
        - action[3]: ask_quantity (0 to max quantity)

        steps:
        1. cancel old/unfilled orders (if placing new orders or orders too old)
        2. place new limit orders based on action
        3. advance market simulator
        4. calculate reward
        5. check done conditions
        """
        # Update order ages for all open orders
        for order_id in list(self.simulator.portfolio.open_orders.keys()):
            if order_id in self.order_ages:
                self.order_ages[order_id] += 1
            else:
                self.order_ages[order_id] = 1
        
       
        max_order_age = 30 
        orders_to_cancel = []
        for order_id, order in list(self.simulator.portfolio.open_orders.items()):
            age = self.order_ages.get(order_id, 0)
            if age > max_order_age and order.filled_quantity == 0:
                orders_to_cancel.append(order_id)
        
        for order_id in orders_to_cancel:
            self.simulator.cancel_agent_order(order_id)
            if order_id in self.order_ages:
                del self.order_ages[order_id]

        #interpret action
        mid_price = self.simulator.orderbook.get_mid_price()
        if mid_price is None:
            #no data, skip
            observation = self._get_observation()
            reward = 0.0
            terminated = False
            truncated = True
            info = self._get_info()
            return observation, reward, terminated, truncated, info

        
        #convert action to prices and quantities
        bid_offset = int(action[0]) -10 # -10 to +10 ticks from mid
        bid_qty_idx = int(action[1])
        ask_offset = int(action[2]) - 10
        ask_qty_idx = int(action[3])

       
        spread = self.simulator.orderbook.get_spread()
        if spread and spread > 0:
            spread_ticks = spread / self.price_tick_size
            # Clamp offsets to be within reasonable range (max 2x spread)
            max_offset_ticks = min(10, max(5, int(spread_ticks * 2)))
            bid_offset = max(-max_offset_ticks, min(max_offset_ticks, bid_offset))
            ask_offset = max(-max_offset_ticks, min(max_offset_ticks, ask_offset))
        else:
           
            bid_offset = max(-5, min(5, bid_offset))
            ask_offset = max(-5, min(5, ask_offset))

        
        bid_price = mid_price + (bid_offset * self.price_tick_size)
        bid_quantity = (bid_qty_idx / self.n_quantity_levels) * self.max_quantity
        ask_price = mid_price + (ask_offset * self.price_tick_size)
        ask_quantity = (ask_qty_idx / self.n_quantity_levels) * self.max_quantity

        #round prices to tick size
        bid_price = round(bid_price/self.price_tick_size) * self.price_tick_size
        ask_price = round(ask_price/self.price_tick_size) * self.price_tick_size
        bid_quantity = round(bid_quantity, self.quantity_precision)
        ask_quantity = round(ask_quantity, self.quantity_precision)

       
        if bid_quantity > 0:
            # Cancel existing buy orders
            for order_id, order in list(self.simulator.portfolio.open_orders.items()):
                if order.side == OrderSide.BUY:
                    self.simulator.cancel_agent_order(order_id)
                    if order_id in self.order_ages:
                        del self.order_ages[order_id]
            
            bid_order = LimitOrder(
                symbol = self.symbol,
                side = OrderSide.BUY,
                price = bid_price,
                quantity = bid_quantity
            )
            try:
                self.simulator.place_agent_order(bid_order)
                # initialize age tracking for new order
                self.order_ages[bid_order.order_id] = 0
            except ValueError as e:
                #not enough liquidity, skip
                pass
        

        if ask_quantity > 0:
            
            for order_id, order in list(self.simulator.portfolio.open_orders.items()):
                if order.side == OrderSide.SELL:
                    self.simulator.cancel_agent_order(order_id)
                    if order_id in self.order_ages:
                        del self.order_ages[order_id]
            
            ask_order = LimitOrder(
                symbol = self.symbol,
                side = OrderSide.SELL,
                price = ask_price,
                quantity = ask_quantity
            )
            try:
                self.simulator.place_agent_order(ask_order)
                self.order_ages[ask_order.order_id] = 0
            except ValueError as e:
                #not enough position, skip
                pass

        #advance market
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        has_more = loop.run_until_complete(self.simulator.step())
        
        # Clean up order ages for filled/cancelled orders
        for order_id in list(self.order_ages.keys()):
            if order_id not in self.simulator.portfolio.open_orders:
                del self.order_ages[order_id]
        
        #reward
        reward = self._calculate_reward()

        #track price history
        if mid_price:
            self.price_history.append(mid_price)
            if len(self.price_history) > self.lookback_window:
                self.price_history.pop(0)
        
        #check done
        self.current_step += 1
        terminated = not has_more #end of data
        truncated = self.current_step >= self.max_steps

        #stop loss / take profit (optional)
        portfolio_stats = self.simulator.portfolio.get_stats(current_price = mid_price)
        if portfolio_stats['total_value'] < self.initial_cash * 0.2:  # Tightened to 20% for earlier intervention
            terminated = True

            reward = max(reward, -10.0)
        
      
        reward = max(-50.0, min(50.0, reward))
        
        observation = self._get_observation()
        info = self._get_info()
        
        if terminated or truncated:
            info['episode_reward'] = reward  # Final reward is already clipped
            
        return observation, reward, terminated, truncated, info
    

    def _get_observation(self) -> np.ndarray:
        """
        convert market state to observation vector

        observation includes:
        order book features
        depth levels
        portfolio state
        market features
        """

        state = self.simulator.get_current_state()
        mid_price = state['mid_price'] or 0.0

        #order book features
        features = [
            state['best_bid'] or 0.0,
            state['best_ask'] or 0.0,
            mid_price,
            state['spread'] or 0.0,
            state['spread_pct'] or 0.0,
            state['imbalance'] or 0.0
        ]

        #depth levels
        bid_levels, ask_levels = state['depth']
        for i in range(10):
            if i < len(bid_levels):
                features.extend([bid_levels[i][0], bid_levels[i][1]])
            else:
                features.extend([0.0, 0.0])
            if i < len(ask_levels):
                features.extend([ask_levels[i][0], ask_levels[i][1]])
            else:
                features.extend([0.0, 0.0])

        #portfolio features
        portfolio = state['portfolio']
        features.extend([
            portfolio['cash'],
            portfolio['position'],
            portfolio.get('unrealized_pnl', 0.0),
            portfolio.get('return_pct', 0.0)
        ])

        #market features
        volatility = self._calculate_volatility()
        price_change = self._calculate_price_change()
        features.extend([volatility, price_change])

        #normalize features
        #TODO: implement normalization
        obs = np.array(features, dtype=np.float32)
        return obs
    

    def _calculate_reward(self) -> float:
        """
        calculate reward for RL agent

        reward = realized_pnl + alpha * unrealized_pnl - beta * inventory_penalty + reward_shaping

        why this?
        realized pnl is the actual profit from filled orders
        unrealized pnl is lower weight mark to market value
        penalize large positions (RM)
        reward shaping: encourage placing orders near spread to get fills
        """
        mid_price = self.simulator.orderbook.get_mid_price()
        if mid_price is None:
            return 0.0
        
        #get portfolio stats
        portfolio_stats = self.simulator.portfolio.get_stats(current_price = mid_price)

      
        realized_pnl = (portfolio_stats['realized_pnl'] / self.initial_cash) * 10

        
        alpha = 0.05  
        unrealized_pnl = (portfolio_stats.get('unrealized_pnl', 0.0) / self.initial_cash) * 10

        
        max_position_value = self.initial_cash * 0.5  
        position_value = abs(portfolio_stats['position'] * mid_price)
        
        beta = 0.1  
        if position_value > max_position_value:
            excess = position_value - max_position_value
            inventory_penalty = (max_position_value / self.initial_cash) * 10 + (excess / self.initial_cash) * 20
        else:
            inventory_penalty = (position_value / self.initial_cash) * 10

        reward = realized_pnl + (alpha * unrealized_pnl) - (beta * inventory_penalty)

        spread = self.simulator.orderbook.get_spread()
        best_bid = self.simulator.orderbook.get_best_bid()
        best_ask = self.simulator.orderbook.get_best_ask()
        
        if spread and spread > 0 and best_bid and best_ask:
            placement_reward = 0.0
            spread_ticks = spread / self.price_tick_size
            
            for order in self.simulator.portfolio.open_orders.values():
                if hasattr(order, 'side') and hasattr(order, 'price'):
                    if order.side.value == 'BUY':
                        distance_ticks = abs(order.price - best_bid) / self.price_tick_size
                        if order.price >= best_bid:
                            placement_reward += 0.001  # Reduced to match new scaling (10x instead of 100x)
                        elif distance_ticks <= 3:
                            placement_reward += 0.0005 * (1 - distance_ticks / 3)  # Reduced scaling
                    else:  # SELL
                        distance_ticks = abs(order.price - best_ask) / self.price_tick_size
                        if order.price <= best_ask:
                            placement_reward += 0.001  # Reduced to match new scaling
                        elif distance_ticks <= 3:
                            # Within 3 ticks of best ask - decreasing reward
                            placement_reward += 0.0005 * (1 - distance_ticks / 3)  # Reduced scaling
            

            if len(self.simulator.portfolio.open_orders) >= 2:
                has_bid = any(o.side.value == 'BUY' for o in self.simulator.portfolio.open_orders.values())
                has_ask = any(o.side.value == 'SELL' for o in self.simulator.portfolio.open_orders.values())
                if has_bid and has_ask:
                    placement_reward += 0.0001  
            
            reward += placement_reward
        

        unfilled_penalty = 0.0
        for order_id, order in self.simulator.portfolio.open_orders.items():
            if order.filled_quantity == 0:
                age = self.order_ages.get(order_id, 0)
               
                if age > 20:
                    penalty = min(0.001 * (age - 20), 0.01)  ]
                    unfilled_penalty += penalty
        
        reward -= unfilled_penalty
        

        reward = max(-10.0, min(10.0, reward))

        return float(reward)

    

    def _calculate_volatility(self) -> float:
        """
        calculate price volatility over lookback window
        """
        if len(self.price_history) < 2:
            return 0.0
        returns = np.diff(self.price_history) / self.price_history[:-1]
        volatility = np.std(returns) if len(returns) > 1 else 0.0
        return float(volatility)
    
    def _calculate_price_change(self) -> float:
        """
        calculate price change over lookback window (momentum)
        """
        if len(self.price_history) < 2:
            return 0.0

        change = (self.price_history[-1] - self.price_history[0]) / self.price_history[0]
        return float(change)
    
    def _get_info(self) -> Dict:
        """
        get additional info for monitoring / debugging
        """
        mid_price = self.simulator.orderbook.get_mid_price()
        portfolio_stats = self.simulator.portfolio.get_stats(current_price = mid_price)
        
        # Diagnostic logging: track order placement and fill statistics
        open_orders = list(self.simulator.portfolio.open_orders.values())
        order_stats = {
            'num_orders': len(open_orders),
            'num_filled': sum(1 for o in open_orders if o.filled_quantity > 0),
            'num_partial': sum(1 for o in open_orders if o.status.value == 'PARTIALLY_FILLED'),
            'avg_distance_from_mid': 0.0,
            'avg_order_age': 0.0,
            'orders_near_spread': 0
        }
        
        if open_orders and mid_price:
            distances = []
            ages = []
            spread = self.simulator.orderbook.get_spread() or 0.0
            best_bid = self.simulator.orderbook.get_best_bid() or mid_price
            best_ask = self.simulator.orderbook.get_best_ask() or mid_price
            
            for order in open_orders:
                if hasattr(order, 'price'):
                    distance = abs(order.price - mid_price)
                    distances.append(distance)
                    
                    # Check if order is near spread (within 3 ticks)
                    if order.side.value == 'BUY':
                        if abs(order.price - best_bid) <= 3 * self.price_tick_size:
                            order_stats['orders_near_spread'] += 1
                    else:  # SELL
                        if abs(order.price - best_ask) <= 3 * self.price_tick_size:
                            order_stats['orders_near_spread'] += 1
                
                # Track order age
                order_id = order.order_id
                if order_id in self.order_ages:
                    ages.append(self.order_ages[order_id])
            
            if distances:
                order_stats['avg_distance_from_mid'] = float(np.mean(distances))
            if ages:
                order_stats['avg_order_age'] = float(np.mean(ages))
        
        return {
            'step': self.current_step,
            'portfolio': portfolio_stats,
            'market': {
                'mid_price': mid_price,
                'spread': self.simulator.orderbook.get_spread(),
                'best_bid': self.simulator.orderbook.get_best_bid(),
                'best_ask': self.simulator.orderbook.get_best_ask(),
            },
            'orders': order_stats  # Diagnostic information
        }
    
    def render(self, mode: str = 'human'):
        """
        render the current state of the environment
        
        Args:
            mode: rendering mode ('human' for console output, 'rgb_array' for image)
        """
        if mode == 'human':
            mid_price = self.simulator.orderbook.get_mid_price()
            portfolio_stats = self.simulator.portfolio.get_stats(current_price=mid_price)
            state = self.simulator.get_current_state()
            
            print("\n" + "="*60)
            print(f"Step: {self.current_step}/{self.max_steps}")
            print("="*60)
            print(f"Market State:")
            print(f"  Symbol: {self.symbol}")
            print(f"  Best Bid: {state['best_bid']:.2f}")
            print(f"  Best Ask: {state['best_ask']:.2f}")
            print(f"  Mid Price: {mid_price:.2f}" if mid_price else "  Mid Price: N/A")
            print(f"  Spread: {state['spread']:.2f}" if state['spread'] else "  Spread: N/A")
            print(f"  Spread %: {state['spread_pct']:.4f}%" if state['spread_pct'] else "  Spread %: N/A")
            print(f"  Imbalance: {state['imbalance']:.4f}" if state['imbalance'] else "  Imbalance: N/A")
            print(f"\nPortfolio:")
            print(f"  Cash: ${portfolio_stats['cash']:.2f}")
            print(f"  Position: {portfolio_stats['position']:.4f}")
            print(f"  Total Value: ${portfolio_stats.get('total_value', 0):.2f}")
            print(f"  Realized PnL: ${portfolio_stats['realized_pnl']:.2f}")
            print(f"  Unrealized PnL: ${portfolio_stats.get('unrealized_pnl', 0):.2f}")
            print(f"  Total PnL: ${portfolio_stats.get('total_pnl', 0):.2f}")
            print(f"  Return %: {portfolio_stats.get('return_pct', 0):.2f}%")
            print(f"  Open Orders: {portfolio_stats['open_orders_count']}")
            print(f"  Total Fills: {portfolio_stats['total_fills']}")
            print(f"\nMarket Features:")
            print(f"  Volatility: {self._calculate_volatility():.6f}")
            print(f"  Price Change: {self._calculate_price_change():.4f}")
            print("="*60 + "\n")
            
        elif mode == 'rgb_array':
            # For future: return numpy array for visualization
            return None
        else:
            raise ValueError(f"Unsupported render mode: {mode}")