"""
Hyperparameter optimization configuration for Nautilus Trading Platform.

This module provides configuration classes for Optuna-based hyperparameter optimization.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class SamplerConfig(BaseModel):
    """
    Configuration for Optuna sampler.

    Samplers determine how to explore the hyperparameter search space.

    Attributes:
        sampler_type: Type of sampler ('tpe', 'random', 'grid', 'cmaes', 'nsgaii').
        seed: Random seed for reproducibility (default: None).
        n_startup_trials: Number of random trials before using sampler (default: 10).
        multivariate: Enable multivariate sampling for TPE (default: True).
        constant_liar: Enable constant liar for parallel TPE (default: False).
    """

    sampler_type: str = Field(default="tpe", description="Sampler type")
    seed: Optional[int] = Field(default=None, description="Random seed")
    n_startup_trials: int = Field(default=10, ge=0, description="Startup trials")
    multivariate: bool = Field(default=True, description="Multivariate sampling (TPE)")
    constant_liar: bool = Field(default=False, description="Constant liar (parallel TPE)")

    @field_validator("sampler_type")
    @classmethod
    def validate_sampler_type(cls, v: str) -> str:
        """Validate sampler type is supported."""
        supported = ["tpe", "random", "grid", "cmaes", "nsgaii", "qmc"]
        if v.lower() not in supported:
            raise ValueError(f"Sampler must be one of {supported}, got: {v}")
        return v.lower()


class PrunerConfig(BaseModel):
    """
    Configuration for Optuna pruner.

    Pruners stop unpromising trials early to save computation time.

    Attributes:
        pruner_type: Type of pruner ('median', 'percentile', 'hyperband', 'threshold', 'none').
        n_startup_trials: Number of trials before pruning starts (default: 5).
        n_warmup_steps: Number of steps before pruning is considered (default: 0).
        interval_steps: Interval of steps for pruning (default: 1).
        percentile: Percentile for percentile pruner (default: 25.0).
        threshold: Threshold value for threshold pruner (optional).
        min_resource: Minimum resource for hyperband (default: 1).
        max_resource: Maximum resource for hyperband (default: 100).
        reduction_factor: Reduction factor for hyperband (default: 3).
    """

    pruner_type: str = Field(default="median", description="Pruner type")
    n_startup_trials: int = Field(default=5, ge=0, description="Startup trials before pruning")
    n_warmup_steps: int = Field(default=0, ge=0, description="Warmup steps")
    interval_steps: int = Field(default=1, ge=1, description="Pruning interval")
    percentile: float = Field(default=25.0, ge=0, le=100, description="Percentile for pruner")
    threshold: Optional[float] = Field(default=None, description="Threshold for pruner")
    min_resource: int = Field(default=1, ge=1, description="Min resource (Hyperband)")
    max_resource: int = Field(default=100, ge=1, description="Max resource (Hyperband)")
    reduction_factor: int = Field(default=3, ge=2, description="Reduction factor (Hyperband)")

    @field_validator("pruner_type")
    @classmethod
    def validate_pruner_type(cls, v: str) -> str:
        """Validate pruner type is supported."""
        supported = ["median", "percentile", "hyperband", "threshold", "none"]
        if v.lower() not in supported:
            raise ValueError(f"Pruner must be one of {supported}, got: {v}")
        return v.lower()


class ObjectiveConfig(BaseModel):
    """
    Configuration for optimization objective.

    Attributes:
        metric: Metric to optimize ('sharpe_ratio', 'total_return', 'sortino_ratio',
                'profit_factor', 'calmar_ratio', 'custom').
        direction: Optimization direction ('maximize' or 'minimize').
        custom_metric_function: Name of custom metric function (if metric='custom').
        constraints: List of constraints (e.g., ['min_trades >= 100', 'win_rate > 0.5']).
        multi_objective: Enable multi-objective optimization (default: False).
        objectives: List of objectives for multi-objective (e.g., ['sharpe_ratio', 'max_drawdown']).
    """

    metric: str = Field(default="sharpe_ratio", description="Metric to optimize")
    direction: str = Field(default="maximize", description="Optimization direction")
    custom_metric_function: Optional[str] = Field(default=None, description="Custom metric function")
    constraints: List[str] = Field(default_factory=list, description="Optimization constraints")
    multi_objective: bool = Field(default=False, description="Multi-objective optimization")
    objectives: List[str] = Field(default_factory=list, description="Multiple objectives")

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, v: str) -> str:
        """Validate metric is supported."""
        supported = [
            "sharpe_ratio",
            "sortino_ratio",
            "total_return",
            "profit_factor",
            "calmar_ratio",
            "max_drawdown",
            "win_rate",
            "return_over_max_drawdown",
            "custom",
        ]
        if v.lower() not in supported:
            raise ValueError(f"Metric must be one of {supported}, got: {v}")
        return v.lower()

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, v: str) -> str:
        """Validate direction is 'maximize' or 'minimize'."""
        if v.lower() not in ["maximize", "minimize"]:
            raise ValueError(f"Direction must be 'maximize' or 'minimize', got: {v}")
        return v.lower()

    @model_validator(mode="after")
    def validate_multi_objective(self) -> "ObjectiveConfig":
        """Validate multi-objective configuration."""
        if self.multi_objective and len(self.objectives) < 2:
            raise ValueError(
                "Multi-objective optimization requires at least 2 objectives, "
                f"got {len(self.objectives)}"
            )
        return self


class ValidationConfig(BaseModel):
    """
    Configuration for optimization validation.

    Attributes:
        enable_validation: Enable out-of-sample validation (default: True).
        validation_method: Validation method ('walk_forward', 'holdout', 'cv').
        train_ratio: Ratio of data for training (default: 0.7).
        n_folds: Number of folds for cross-validation (default: 5).
        walk_forward_windows: Number of windows for walk-forward (default: 5).
        anchored: Use anchored walk-forward (default: False).
        refit: Refit on all data after validation (default: False).
    """

    enable_validation: bool = Field(default=True, description="Enable validation")
    validation_method: str = Field(default="walk_forward", description="Validation method")
    train_ratio: float = Field(default=0.7, ge=0.1, le=0.9, description="Training ratio")
    n_folds: int = Field(default=5, ge=2, le=10, description="CV folds")
    walk_forward_windows: int = Field(default=5, ge=2, description="Walk-forward windows")
    anchored: bool = Field(default=False, description="Anchored walk-forward")
    refit: bool = Field(default=False, description="Refit after validation")

    @field_validator("validation_method")
    @classmethod
    def validate_validation_method(cls, v: str) -> str:
        """Validate validation method is supported."""
        supported = ["walk_forward", "holdout", "cv", "none"]
        if v.lower() not in supported:
            raise ValueError(f"Validation method must be one of {supported}, got: {v}")
        return v.lower()


class StorageConfig(BaseModel):
    """
    Configuration for Optuna study storage.

    Attributes:
        storage_type: Storage type ('sqlite', 'postgresql', 'mysql', 'in_memory').
        storage_url: Storage URL (e.g., 'sqlite:///optuna.db').
        study_name: Name of the study (default: auto-generated).
        load_if_exists: Load existing study if found (default: True).
    """

    storage_type: str = Field(default="sqlite", description="Storage type")
    storage_url: Optional[str] = Field(default=None, description="Storage URL")
    study_name: Optional[str] = Field(default=None, description="Study name")
    load_if_exists: bool = Field(default=True, description="Load if exists")

    @field_validator("storage_type")
    @classmethod
    def validate_storage_type(cls, v: str) -> str:
        """Validate storage type is supported."""
        supported = ["sqlite", "postgresql", "mysql", "in_memory"]
        if v.lower() not in supported:
            raise ValueError(f"Storage type must be one of {supported}, got: {v}")
        return v.lower()

    @model_validator(mode="after")
    def set_default_storage_url(self) -> "StorageConfig":
        """Set default storage URL if not provided."""
        if self.storage_url is None:
            if self.storage_type == "sqlite":
                storage_dir = Path("optimization_studies")
                storage_dir.mkdir(parents=True, exist_ok=True)
                self.storage_url = f"sqlite:///{storage_dir}/optuna.db"
            elif self.storage_type == "in_memory":
                self.storage_url = "sqlite:///:memory:"
        return self


class ParallelizationConfig(BaseModel):
    """
    Configuration for parallel optimization.

    Attributes:
        n_jobs: Number of parallel jobs (-1 = all CPUs, 1 = sequential).
        timeout_per_trial: Timeout per trial in seconds (0 = no timeout).
        gc_after_trial: Run garbage collection after each trial (default: True).
        show_progress_bar: Show progress bar during optimization (default: True).
    """

    n_jobs: int = Field(default=1, ge=-1, description="Parallel jobs")
    timeout_per_trial: int = Field(default=0, ge=0, description="Timeout per trial (s)")
    gc_after_trial: bool = Field(default=True, description="GC after trial")
    show_progress_bar: bool = Field(default=True, description="Show progress bar")

    @field_validator("n_jobs")
    @classmethod
    def validate_n_jobs(cls, v: int) -> int:
        """Validate n_jobs is valid."""
        if v == 0:
            raise ValueError("n_jobs cannot be 0. Use -1 for all CPUs or positive integer.")
        return v


class OptimizationConfig(BaseModel):
    """
    Main optimization configuration.

    Attributes:
        name: Name of the optimization run.
        description: Description of the optimization.
        n_trials: Number of optimization trials (default: 100).
        timeout: Total timeout in seconds (0 = no timeout).
        sampler: Sampler configuration.
        pruner: Pruner configuration.
        objective: Objective configuration.
        validation: Validation configuration.
        storage: Storage configuration.
        parallelization: Parallelization configuration.
        hyperparameter_space: Dictionary defining hyperparameter search space.
        output_path: Path to save optimization results.
        save_visualizations: Save Optuna visualization plots (default: True).
        save_best_params: Save best parameters to file (default: True).
    """

    name: str = Field(..., description="Optimization run name")
    description: Optional[str] = Field(default="", description="Optimization description")
    n_trials: int = Field(default=100, ge=1, description="Number of trials")
    timeout: int = Field(default=0, ge=0, description="Total timeout (s, 0=no timeout)")
    sampler: SamplerConfig = Field(default_factory=SamplerConfig, description="Sampler config")
    pruner: PrunerConfig = Field(default_factory=PrunerConfig, description="Pruner config")
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig, description="Objective config")
    validation: ValidationConfig = Field(default_factory=ValidationConfig, description="Validation config")
    storage: StorageConfig = Field(default_factory=StorageConfig, description="Storage config")
    parallelization: ParallelizationConfig = Field(
        default_factory=ParallelizationConfig, description="Parallelization config"
    )
    hyperparameter_space: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Hyperparameter search space"
    )
    output_path: Path = Field(default=Path("optimization_studies"), description="Output path")
    save_visualizations: bool = Field(default=True, description="Save visualizations")
    save_best_params: bool = Field(default=True, description="Save best params")

    @model_validator(mode="after")
    def ensure_output_path(self) -> "OptimizationConfig":
        """Ensure output path exists."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self.model_dump()

    @classmethod
    def from_env(cls, **kwargs) -> "OptimizationConfig":
        """
        Create configuration from environment variables.

        Args:
            **kwargs: Override parameters.

        Returns:
            OptimizationConfig instance.

        Example:
            >>> config = OptimizationConfig.from_env(name="MACD Optimization")
        """
        sampler_config = SamplerConfig(
            sampler_type="tpe",  # Tree-structured Parzen Estimator (best for most cases)
            seed=42,
        )

        pruner_config = PrunerConfig(
            pruner_type="median",  # Median pruner (safe and effective)
            n_startup_trials=10,
        )

        objective_config = ObjectiveConfig(
            metric="sharpe_ratio",  # Default metric
            direction="maximize",
        )

        storage_config = StorageConfig(
            storage_type="sqlite",
            storage_url=os.getenv("DB_URL", "sqlite:///optimization_studies/optuna.db"),
        )

        parallelization_config = ParallelizationConfig(
            n_jobs=int(os.getenv("OPTUNA_N_JOBS", "1")),
            show_progress_bar=True,
        )

        return cls(
            n_trials=int(os.getenv("OPTUNA_N_TRIALS", "100")),
            timeout=int(os.getenv("OPTUNA_TIMEOUT", "0")),
            sampler=sampler_config,
            pruner=pruner_config,
            objective=objective_config,
            storage=storage_config,
            parallelization=parallelization_config,
            **kwargs,
        )


# =============================================================================
# EXAMPLE CONFIGURATIONS
# =============================================================================


def create_default_optimization_config() -> OptimizationConfig:
    """
    Create a default optimization configuration.

    Returns:
        Default OptimizationConfig.

    Example:
        >>> config = create_default_optimization_config()
        >>> config.n_trials
        100
    """
    return OptimizationConfig(
        name="Default Optimization",
        description="Default optimization configuration with TPE sampler and median pruner",
        n_trials=100,
        hyperparameter_space={
            "example_param": {
                "type": "float",
                "low": 0.0,
                "high": 1.0,
            }
        },
    )


def create_fast_optimization_config(n_trials: int = 50) -> OptimizationConfig:
    """
    Create a fast optimization configuration for quick testing.

    Args:
        n_trials: Number of trials (default: 50).

    Returns:
        Fast OptimizationConfig.

    Example:
        >>> config = create_fast_optimization_config(n_trials=20)
        >>> config.n_trials
        20
    """
    return OptimizationConfig(
        name="Fast Optimization",
        description="Fast optimization for testing",
        n_trials=n_trials,
        sampler=SamplerConfig(sampler_type="random"),  # Random is fastest
        pruner=PrunerConfig(pruner_type="median", n_startup_trials=5),
        validation=ValidationConfig(enable_validation=False),  # Skip validation for speed
        parallelization=ParallelizationConfig(n_jobs=-1),  # Use all CPUs
    )


def create_thorough_optimization_config(n_trials: int = 500) -> OptimizationConfig:
    """
    Create a thorough optimization configuration for production use.

    Args:
        n_trials: Number of trials (default: 500).

    Returns:
        Thorough OptimizationConfig.

    Example:
        >>> config = create_thorough_optimization_config()
        >>> config.n_trials
        500
    """
    return OptimizationConfig(
        name="Thorough Optimization",
        description="Thorough optimization with walk-forward validation",
        n_trials=n_trials,
        sampler=SamplerConfig(sampler_type="tpe", n_startup_trials=20, multivariate=True),
        pruner=PrunerConfig(pruner_type="hyperband"),
        objective=ObjectiveConfig(metric="sharpe_ratio", direction="maximize"),
        validation=ValidationConfig(
            enable_validation=True,
            validation_method="walk_forward",
            walk_forward_windows=5,
        ),
        parallelization=ParallelizationConfig(n_jobs=4, timeout_per_trial=3600),
        save_visualizations=True,
        save_best_params=True,
    )


if __name__ == "__main__":
    # Example usage
    config = create_default_optimization_config()
    print(config.model_dump_json(indent=2))
