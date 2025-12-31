from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "LiquidityZero"

    #exhcnage config
    EXCHANGE_ID: str="binance"
    SYMBOL: str="BTC/USDT"

    #database config
    REDIS_URL: str = "redis://localhost:6379/0"
    TIMESCALE_URL: str = "postgresql://postgres:password@localhost:5432/liquidity_db"
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()