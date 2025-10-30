
from decimal import Decimal
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.indicators.macd import MovingAverageConvergenceDivergence
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy


class MACDStrategyConfig(StrategyConfig):
    """Configuration pour la stratégie MACD"""
    instrument_id: str
    bar_type: str
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    trade_size: Decimal = Decimal("100")


class MACDStrategy(Strategy):
    """
    Stratégie MACD simple:
    - Achat quand MACD > Signal
    - Vente quand MACD < Signal
    """

    def __init__(self, config: MACDStrategyConfig):
        super().__init__(config)

        # Configuration
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type = config.bar_type
        self.trade_size = config.trade_size

        # Indicateur MACD
        self.macd = MovingAverageConvergenceDivergence(
            fast_period=config.fast_period,
            slow_period=config.slow_period,
            signal_period=config.signal_period,
        )

        # État
        self.position_opened = False

    def on_start(self):
        """Actions au démarrage de la stratégie"""
        self.subscribe_bars(self.bar_type)
        self.log.info(f"Stratégie démarrée pour {self.instrument_id}")

    def on_bar(self, bar: Bar):
        """Appelé à chaque nouvelle barre"""
        # Mettre à jour l'indicateur
        self.macd.handle_bar(bar)

        # Attendre que l'indicateur soit initialisé
        if not self.macd.initialized:
            return

        # Récupérer les valeurs
        macd_value = self.macd.value
        signal_value = self.macd.signal

        # Logique de trading
        if macd_value > signal_value and not self.position_opened:
            # Signal d'achat
            self.buy()

        elif macd_value < signal_value and self.position_opened:
            # Signal de vente
            self.sell()

    def buy(self):
        """Ouvrir une position longue"""
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(self.trade_size),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
        self.position_opened = True
        self.log.info(f"📈 ACHAT: MACD={self.macd.value:.4f}, Signal={self.macd.signal:.4f}")

    def sell(self):
        """Fermer la position"""
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(self.trade_size),
            time_in_force=TimeInForce.GTC,
        )
        self.submit_order(order)
        self.position_opened = False
        self.log.info(f"📉 VENTE: MACD={self.macd.value:.4f}, Signal={self.macd.signal:.4f}")

    def on_stop(self):
        """Actions à l'arrêt de la stratégie"""
        # Fermer toutes les positions ouvertes
        self.close_all_positions(self.instrument_id)
        self.log.info("Stratégie arrêtée")
