from dataclasses import dataclass, field
from typing import Union, TypeAlias


_default_lambda_range = (5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4)


def _as_lambda_tuple(value):
    """Normalize a legacy lambda_range to a hashable tuple.

    ModelConfig is passed to jax as a static argument, so every field has to be
    hashable -- a list (which legacy callers pass) raises at trace time with
    "Non-hashable static arguments are not supported". None falls through to the
    module default rather than tripping the explicit raise in gc_extraction.
    """
    if value is None:
        return _default_lambda_range
    if isinstance(value, (int, float)):
        return (float(value),)
    return tuple(value)


@dataclass(frozen=True)
class ModelSerialConfig:
    """
    run models serially on a single CPU thread.
    """
    pass

@dataclass(frozen=True)
class ModelVmapConfig:
    """
    vectorize models using jax.vmap on a single device. used for GPU
    acceleration.
    """
    pass


@dataclass(frozen=True)
class ModelShardConfig:
    """
    parallelize models across JAX devices using shard_map (or pmap). if
    utilizing sharding to parallelize across multiple cpus on a machine with
    performance and efficiency cores (e.g., most modern macs) you MUST specify
    n_devices to be the exactly the number of performance cores and NOT the
    total cpu count, else every batched computation will be blocked until the
    efficiency cores finish, bottlenecking the GC testing.
    """
    n_devices: int = 1


@dataclass(frozen=True)
class ModelMultiprocessConfig:
    """
    parallelize models using independent Python worker processes.
    """
    n_workers: int = 1

ModelParallelConfig: TypeAlias = (
    ModelSerialConfig
    | ModelVmapConfig
    | ModelShardConfig
    | ModelMultiprocessConfig
)


@dataclass(frozen=True)
class ModelLatentConfig:
    order: int = 2
    n_eigenmodes: int = 4
    n_orients: int = 1
    n_segments: int = 1

@dataclass(frozen=True)
class ModelSparsityConfig:
    self_history : Union[int, None] = None
    alpha: float = 0.0
    beta: float = 0.0
    var_thr:float = 1.0
    sparsity_factor: float = 0.0
    lambda_range: Union[float, tuple[float]] = field(default_factory=lambda: \
                                                     _default_lambda_range)
    lambda1: Union[float, None] = None
    lambda2: Union[float, None] = None
    negligible_candidate_link_energy_thr: float = 0.99

@dataclass(frozen=True)
class ModelForwardConfig:
    loose: float = 0.0
    depth: float = 0.0
    rank: Union[int, None] = None
    pca: bool = True
    patch_idx: tuple[int] = field(default_factory=tuple)

@dataclass(frozen=True)
class ModelOptimizerConfig:
    max_iter: int = field(default=500, metadata={"static": True})  
    n_warmup_iter: int = 25   # denotes how many blas iters before using jax 
    max_cyclic_iter: int = 3  # legacy; deprecated
    tol: float = 1e-8
    warm_start: bool = False
    
@dataclass(frozen=True)
class ModelValidationConfig:
    cv: int = 5
    use_es: bool = False
    cv_type: str = 'seeded'

@dataclass(frozen=True)
class ModelNumericalConfig:
    use_lapack: bool = True
    verbose: bool = False


@dataclass(frozen=True)
class ModelConfig:
    latent: ModelLatentConfig
    sparsity: ModelSparsityConfig
    forward: ModelForwardConfig
    optimizer: ModelOptimizerConfig
    validation: ModelValidationConfig
    numerical: ModelNumericalConfig
    parallel: ModelParallelConfig

    @classmethod
    def from_legacy_kwargs(cls, kwargs):
        parallel_mode = kwargs.pop("parallel_mode", "serial")

        if parallel_mode == "serial":
            parallel = ModelSerialConfig()

        elif parallel_mode == "vmap":
            parallel = ModelVmapConfig()

        elif parallel_mode == "shard":
            parallel = ModelShardConfig(
                n_devices=kwargs.pop("n_devices")
            )

        elif parallel_mode == "multiprocess":
            parallel = ModelMultiprocessConfig(n_workers=kwargs.pop("n_workers"))

        else:
            raise ValueError(f"Unknown parallel_mode: {parallel_mode}")
        
        return cls(
            parallel = parallel,
            latent = ModelLatentConfig(
                order = kwargs.pop("order"),
                n_eigenmodes = kwargs.pop("n_eigenmodes", 2),
                n_orients = kwargs.pop("n_orients", 1),
                n_segments = kwargs.pop("n_segments", 1),
            ),

            sparsity = ModelSparsityConfig(
                self_history = kwargs.pop("self_history", None),
                alpha = kwargs.pop("alpha", 0.0),
                beta = kwargs.pop("beta", 0.0),
                var_thr = kwargs.pop("var_thr", 1.0),
                sparsity_factor = kwargs.pop("sparsity_factor", 0.0),
                lambda_range = _as_lambda_tuple(
                                kwargs.pop("lambda_range", None)),
                negligible_candidate_link_energy_thr = \
                    kwargs.pop("negligible_candidate_link_energy_thr", 0.99),
            ),

            forward = ModelForwardConfig(
                loose = kwargs.pop("loose", 0.0),
                depth = kwargs.pop("depth", 0.0),
                rank = kwargs.pop("rank", None),
                pca = kwargs.pop("pca", True),
                patch_idx = kwargs.pop("patch_idx", ()),
            ),

            optimizer = ModelOptimizerConfig(
                max_iter = kwargs.pop("max_iter", 500),
                max_cyclic_iter = kwargs.pop("max_cyclic_iter", 3),
                n_warmup_iter= kwargs.pop("n_warmup_iter", 25),
                tol = kwargs.pop("tol", 1e-8),
                warm_start = kwargs.pop("warm_start", False),
            ),
            
            validation = ModelValidationConfig(
                cv = kwargs.pop("cv", 5),
                cv_type = kwargs.pop("cv_type", "seeded"),
                use_es = kwargs.pop("use_es", True),
            ),
            
            numerical = ModelNumericalConfig(
                use_lapack = kwargs.pop("use_lapack", True),
                verbose = kwargs.pop("verbose", False),
            ),
        )