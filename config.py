"""
Centralized configuration management for the Autonomous Trading System.
All critical parameters and environment variables are defined here.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DataConfig:
    """Configuration for data ingestion and processing."""
    # Data sources
    DATA_SOURCES: List[str] = ["binance", "kraken", "coinbase"]
    TIME_INTERVALS: List[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
    
    # Feature engineering
    TECHNICAL_INDICATORS: List[str] = ["RSI", "MACD", "BBANDS", "ATR", "OBV"]
    LOOKBACK_WINDOW: int = 100
    SEQ_LENGTH: int = 50  # For sequence modeling
    
    # Data storage
    FIREBASE_COLLECTION: str = "market_data"
    CACHE_TTL_SECONDS: int = 300  # 5 minutes

@dataclass
class ModelConfig:
    """Configuration for SSL and RL models."""
    # Self-Supervised Learning
    SSL_HIDDEN_DIM: int = 128
    SSL_DROPOUT: float = 0.2
    SSL_LEARNING_RATE: float = 0.001
    SSL_BATCH_SIZE: int = 64
    SSL_PRETRAIN_EPOCHS: int = 100
    
    # Reinforcement Learning
    RL_ENV_NAME: str = "TradingEnv-v0"
    RL_LEARNING_RATE: float = 0.0003
    RL_GAMMA: float = 0.99
    RL_BUFFER_SIZE: int = 100000
    RL_BATCH_SIZE: int = 256
    RL_TARGET_UPDATE_FREQ: int = 100
    
    # Model storage
    MODEL_BUCKET: str = "trading-models"
    CHECKPOINT_FREQ: int = 1000  # Save every 1000 steps

@dataclass
class TradingConfig:
    """Configuration for trading execution."""
    # Risk management
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio per trade
    STOP_LOSS_PCT: float = 0.02  # 2% stop loss
    TAKE_PROFIT_PCT: float = 0.05  # 5% take profit
    MAX_DRAWDOWN_PCT: float = 0.15  # 15% max drawdown
    
    # Execution
    ORDER_TIMEOUT_SECONDS: int = 30
    RETRY_ATTEMPTS: int = 3
    
    # Portfolio
    INITIAL_CAPITAL: float = 100000.0
    ALLOCATION_STRATEGY: str = "risk_parity"

@dataclass
class FirebaseConfig:
    """Firebase configuration."""
    CREDENTIALS_PATH: str = os.getenv("FIREBASE_CREDENTIALS_PATH", "./firebase-credentials.json")
    DATABASE_URL: str = os.getenv("FIREBASE_DATABASE_URL", "")
    COLLECTION_PREFIX: str = "trading_system_"
    
    @property
    def state_collection(self) -> str:
        return f"{self.COLLECTION_PREFIX}state"
    
    @property
    def trades_collection(self) -> str:
        return f"{self.COLLECTION_PREFIX}trades"
    
    @property
    def metrics_collection(self) -> str:
        return f"{self.COLLECTION_PREFIX}metrics"

@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    LOG_LEVEL: str = "INFO"
    METRICS_UPDATE_INTERVAL: int = 60  # seconds
    TELEGRAM_ALERT_THRESHOLD: float = 0.1  # 10% drawdown alert
    
    # Performance thresholds
    MAX_LATENCY_MS: int = 1000
    MIN_DATA_QUALITY_SCORE: float = 0.8

class Config:
    """Main configuration class aggregating all configs."""
    def __init__(self):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.trading = TradingConfig()
        self.firebase = FirebaseConfig()
        self.monitoring = MonitoringConfig()
        
        # Validate critical environment variables
        self._validate_env_vars()
    
    def _validate_env_vars(self) -> None:
        """Validate required environment variables."""
        required_vars = ["FIREBASE_CREDENTIALS_PATH", "FIREBASE_DATABASE_URL"]
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            raise ValueError(f"Missing required environment variables: