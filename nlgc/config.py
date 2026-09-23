from dataclasses import dataclass, field, fields
from multiprocessing import cpu_count
from typing import Union, TypeAlias
import subprocess


_default_lambda_range = (5e-1, 2e-1, 1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4)


def _default_n_workers():
    """P-core count on Apple silicon, else total cpu count.

    Each worker carries its own JAX runtime, so the E-cores cost more in memory
    and scheduler pressure than they return. Anywhere the sysctl key is missing
    (Linux, Intel macs) int() raises and we fall back.
    """
    try:
        return int(subprocess.run(['sysctl', '-n', 'hw.perflevel0.logicalcpu'],
                                  capture_output=True, text=True).stdout)
    except (OSError, ValueError):
        num_cores = cpu_count()
        print("could not identify P-core count, possibly because you are",
              "running on a linux machine or on an older mac that doesn't", 
              "expose the number of Performance cores. ",
              f"defaulting to {num_cores}")
        return num_cores


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

    unlike the shard/vmap paths this runs the full two-phase EM (em_blas warmup
    then em_jax) on each reduced model, at the cost of one JAX runtime and one
    em_jax compilation per worker.

    n_workers should be the number of PERFORMANCE cores, not the total cpu
    count: each worker carries its own JAX runtime, and on machines with
    efficiency cores (most modern macs) the extra workers cost more in memory
    and scheduler pressure than they return.
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
    lagsparsity: bool = True
    self_history : Union[int, None] = None
    alpha: float = 0.0
    beta: float = 0.0
    var_thr:float = 1.0
    sparsity_factor: float = 0.0
    lambda_range: Union[float, tuple[float]] = field(default_factory=lambda: \
                                                     _default_lambda_range)
    lambda1: Union[float, None] = None
    lambda2: Union[float, None] = None
    negligible_candidate_link_energy_thr: float = 1.0
    # Bool to use wald screen or not
    use_wald_screen: bool = True
    # Alpha for wald screen
    wald_screen_alpha: float = 0.05
    use_empirical_null: bool = False

@dataclass(frozen=True)
class ModelQPriorConfig:
    lkj_mode: bool = False
    eta: float = 1
    nu0: Union[int, None] = None
    q_base: float = 1e-4
    # source_mass: Union[, None] = None
    sigma_gamma: float = 1
    sigma_min: float = .25
    sigma_max: float = 4.0
    eig_floor: float = 1e-10

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
    max_cyclic_iter: int = 3  
    max_fasta_iter: int = 1000
    tol: float = 1e-4
    fasta_tol: float = 1e-5
    warm_start: bool = False
    
@dataclass(frozen=True)
class ModelValidationConfig:
    cv: int = 5
    use_es: bool = False
    cv_type: str = "DisturbanceCV"

@dataclass(frozen=True)
class ModelNumericalConfig:
    use_lapack: bool = True
    verbose: bool = False

@dataclass(frozen=True)
class ModelDebugConfig:
    verbose: bool = False
    plotlevel: int = 0

@dataclass(frozen=True)
class ModelGCTestConfig:
    gc_test_method: str = "likelihood ratio"

@dataclass(frozen=True)
class ModelConfig:
    latent: ModelLatentConfig
    sparsity: ModelSparsityConfig
    qprior: ModelQPriorConfig
    forward: ModelForwardConfig
    optimizer: ModelOptimizerConfig
    validation: ModelValidationConfig
    numerical: ModelNumericalConfig
    parallel: ModelParallelConfig
    gctest: ModelGCTestConfig
    debug: ModelDebugConfig

    @classmethod
    def from_legacy_kwargs(cls, kwargs):
        parallel_mode = kwargs.pop("parallel_mode", "serial")

        if parallel_mode == "serial":
            parallel = ModelSerialConfig()

        elif parallel_mode == "vmap":
            parallel = ModelVmapConfig()

        elif parallel_mode == "shard":
            parallel = ModelShardConfig(
                n_devices=kwargs.pop("n_devices", 1)
            )

        elif parallel_mode == "multiprocess":
            n_workers = kwargs.pop("n_workers", None)
            if n_workers is None or n_workers <= 0:
                n_workers = _default_n_workers()
            parallel = ModelMultiprocessConfig(n_workers=n_workers)

        else:
            raise ValueError(f"Unknown parallel_mode: {parallel_mode}")
        
        verbose = kwargs.pop("verbose", False)

        return cls(
            parallel = parallel,

            latent = ModelLatentConfig(
                order = kwargs.pop("order"),
                n_eigenmodes = kwargs.pop("n_eigenmodes", 2),
                n_orients = kwargs.pop("n_orients", 1),
                n_segments = kwargs.pop("n_segments", 1),
            ),

            sparsity = ModelSparsityConfig(
                lagsparsity = kwargs.pop("lagsparsity", True),
                self_history = kwargs.pop("self_history", None),
                alpha = kwargs.pop("alpha", 0.0),
                beta = kwargs.pop("beta", 0.0),
                var_thr = kwargs.pop("var_thr", 1.0),
                sparsity_factor = kwargs.pop("sparsity_factor", 0.0),
                lambda_range = _as_lambda_tuple(
                                kwargs.pop("lambda_range", None)),
                negligible_candidate_link_energy_thr = \
                    kwargs.pop("negligible_candidate_link_energy_thr", 1.0),
                lambda1 = kwargs.pop("lambda1", None),
                lambda2 = kwargs.pop("lambda2", None),
                use_wald_screen = kwargs.pop("use_wald_screen", True),
                wald_screen_alpha = kwargs.pop("wald_screen_alpha", 0.05),
                use_empirical_null = kwargs.pop("use_empirical_null", False),
            ),

            qprior = ModelQPriorConfig(
                lkj_mode = kwargs.pop("lkj_mode", False),
                eta = kwargs.pop("eta", 1.0),
                nu0 = kwargs.pop("nu0", None),
                q_base = kwargs.pop("q_base", 1e-4),
                sigma_gamma = kwargs.pop("sigma_gamma", 1),
                sigma_min = kwargs.pop("sigma_min", .25),
                sigma_max = kwargs.pop("sigma_max", 4.0),
                eig_floor = kwargs.pop("eig_floor", 1e-10),
            ),

            forward = ModelForwardConfig(
                loose = kwargs.pop("loose", 0.0),
                depth = kwargs.pop("depth", 0.0),
                rank = kwargs.pop("rank", None),
                pca = kwargs.pop("pca", True),
                # jax static arg: every field must stay hashable
                patch_idx = tuple(kwargs.pop("patch_idx", ())),
            ),

            optimizer = ModelOptimizerConfig(
                max_iter = kwargs.pop("max_iter", 500),
                max_cyclic_iter = kwargs.pop("max_cyclic_iter", 3),
                tol = kwargs.pop("tol", 1e-4),
                warm_start = kwargs.pop("warm_start", False),
                max_fasta_iter = kwargs.pop("max_fasta_iter", 1000),
                fasta_tol = kwargs.pop("fasta_tol", 1e-5),
            ),
            
            validation = ModelValidationConfig(
                cv = kwargs.pop("cv", 5),
                cv_type = kwargs.pop("cv_type", "DisturbanceCV"),
                use_es = kwargs.pop("use_es", True),
            ),
            
            numerical = ModelNumericalConfig(
                use_lapack = kwargs.pop("use_lapack", True),
                verbose = verbose,
            ),

            gctest = ModelGCTestConfig(
                gc_test_method = kwargs.pop("gc_test_method", 
                                            "likelihood ratio")
            ),

            debug = ModelDebugConfig(
                verbose = kwargs.pop("debug_verbose", False),  # collides
                plotlevel = kwargs.pop("plotlevel", 0),
            ),
        )


# must track the ModelParallelConfig union
_PARALLEL_MODES = {
    ModelSerialConfig: "serial",
    ModelVmapConfig: "vmap",
    ModelShardConfig: "shard",
    ModelMultiprocessConfig: "multiprocess",
}


def to_legacy_kwargs(config):
    """Flatten a ModelConfig into from_legacy_kwargs' kwargs.

    Writes every field: several from_legacy_kwargs defaults disagree with the
    dataclass ones, so an omitted key would come back changed.
    """
    parallel = config.parallel
    mode = _PARALLEL_MODES.get(type(parallel))
    if mode is None:
        raise ValueError(f"Unrecognized parallel config: {type(parallel)}")

    kwargs = {"parallel_mode": mode}
    for section in (config.latent, config.sparsity, config.qprior, config.forward,
                    config.optimizer, config.validation, config.numerical,
                    config.gctest, parallel):
        for f in fields(section):
            value = getattr(section, f.name)
            kwargs[f.name] = list(value) if isinstance(value, tuple) else value

    # would collide with numerical.verbose
    kwargs["debug_verbose"] = config.debug.verbose
    kwargs["plotlevel"] = config.debug.plotlevel
    return kwargs
