from edustudio.quickstart import run_edustudio

run_edustudio(
    dataset='Bridge2Algebra_0607',
    cfg_file_name=None,
    traintpl_cfg_dict={
        'cls': 'GeneralTrainTPL',
        'decay_function': 'exp'
    },
    datatpl_cfg_dict={
        'cls': 'KTInterCptUnfoldDataTPL'
    },
    modeltpl_cfg_dict={
        'cls': 'AKT',
    },
    evaltpl_cfg_dict={
        'clses': ['PredictionEvalTPL'],
    }
)
