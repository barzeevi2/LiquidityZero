from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "LiquidityZero"

    #exchange config
    EXCHANGE_ID: str="binanceus"  # Options: "binance" or "binanceus" (for US users)
    SYMBOL: str="BTC/USDT"

    #simulator config
    MAKER_FEE_RATE: float = -0.0001 #rebate
    TAKER_FEE_RATE: float = 0.0001 #fee
    DEFAULT_INITIAL_CASH: float = 10000.0 #initial cash
    INITIAL_CASH: float = 10000.0 #initial cash for training (alias for DEFAULT_INITIAL_CASH)
    MAX_POSITION_SIZE: float = 10.0 #maximum BTC position
    SLIPPAGE_MODEL: str = "linear"
    
    #training environment config
    MAX_STEPS: int = 1000 #maximum steps per episode
    PRICE_TICK_SIZE: float = 0.01 #minimum price increment
    QUANTITY_PRECISION: int = 4 #decimal places for quantity
    MAX_QUANTITY: float = 1.0 #maximum order quantity
    N_PRICE_LEVELS: int = 21 #number of price levels in action space
    N_QUANTITY_LEVELS: int = 10 #number of quantity levels in action space
    LOOKBACK_WINDOW: int = 10 #lookback window for market features 

    #database config
    REDIS_URL: str = "redis://localhost:6379/0"
    TIMESCALE_URL: str = "postgresql://postgres:password@localhost:5432/liquidity_db"
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()