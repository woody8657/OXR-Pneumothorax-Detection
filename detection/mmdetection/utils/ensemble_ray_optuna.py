import os
import argparse
import json
from ensemble import ensemble
from cocoeval import evaluate
import datetime
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState

def wrap(trial):
    param = {}
    param['weights1'] = trial.suggest_float("DF_f2", 0.0, 1.0)
    param['weights2'] = trial.suggest_float("DF_f4", 0.0, 1.0)
    param['weights3'] = trial.suggest_float("TO_f7", 0.0, 1.0)
    param['weights4'] = trial.suggest_float("TOP_f2", 0.0, 1.0)
    param['weights5'] = trial.suggest_float("VFP_f8", 0.0, 1.0)
    param['iou_thr'] = trial.suggest_float("IoU", 0.0, 1.0)
    param['method'] = trial.suggest_categorical("Algo", ["nmw", "wbf"])
    output_ensemble = ensemble(opt.gt, opt.jsons, param)
    opt.output_name = str(datetime.datetime.now()) + 'ensemble.json'
    with open(opt.output_name, mode='w') as file:
        json.dump(output_ensemble, file)
    metric = evaluate(opt.gt, opt.output_name)
    os.remove(opt.output_name)
    result = {'ap5': metric[1], 'aps': metric[3]}
    return result['ap5']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', help='ground truth path')
    parser.add_argument('--jsons', nargs='+', type=str)
    parser.add_argument('--output-name', default='output.json', help='output file name')
    global opt
    opt = parser.parse_args()

    from optuna.samplers import MOTPESampler
    study = optuna.create_study(direction='maximize', sampler=MOTPESampler(seed=42))
    # best_80k = {
    #     'DF_f2': 0.72412,
    #     'DF_f4': 0.20912,
    #     'TO_f7': 0.20482,
    #     'TOP_f2': 0.41983,
    #     'VFP_f8': 0.33407,
    #     'IoU' : 0.60445,
    #     'Algo': 'wbf'
    # }
    # study.enqueue_trial(best_80k)
    # study.optimize(wrap,  n_jobs=15, callbacks=[MaxTrialsCallback(90, states=(TrialState.COMPLETE,))])
    study.optimize(wrap, n_trials=100)

    print('Number of finished trials:', len(study.trials))
    print('Best trial parameters:', study.best_trial.params)
    print('Best score:', study.best_value)
    from optuna.visualization import plot_parallel_coordinate, plot_contour, plot_param_importances, plot_optimization_history
    fig = plot_contour(study)
    fig.write_image("optuna/hyp.png")

    fig1 = plot_param_importances(study)
    fig1.write_image("optuna/hyp_importance.png")

    fig2 = plot_optimization_history(study)
    fig2.write_image("optuna/hyp_history.png")

    fig3 = plot_parallel_coordinate(study)
    fig3.write_image("optuna/hyp_comb.png")

    fig4 = plot_contour(study, params=["DF_f2", "IoU"])
    fig4.write_image("optuna/hyp_relative.png")