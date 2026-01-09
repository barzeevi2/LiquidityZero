from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "LiquidityZero"

    #exhcnage config
    EXCHANGE_ID: str="binance"
    SYMBOL: str="BTC/USDT"

    #simulator config
    MAKER_FEE_RATE: float = -0.0001 #rebate
    TAKER_FEE_RATE: float = 0.0001 #fee
    DEFAULT_INITIAL_CASH: float = 10000.0 #initial cash
    MAX_POSITION_SIZE: float = 10.0 #maximum BTC position
    SLIPPAGE_MODEL: str = "linear" 

    #database config
    REDIS_URL: str = "redis://localhost:6379/0"
    TIMESCALE_URL: str = "postgresql://postgres:password@localhost:5432/liquidity_db"
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()