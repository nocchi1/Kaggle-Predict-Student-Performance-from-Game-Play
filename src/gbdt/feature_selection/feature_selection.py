
import pandas as pd
from typing import *
from pathlib import Path
import lightgbm as lgb
import pickle


def lgb_feature_select(models: List[lgb.Booster], use_feat_num: Optional[int]):
    use_feats = []
    for model in models:
        name = model.feature_name()
        score = model.feature_importance(importance_type='gain')
        feat_df = pd.DataFrame(dict(score = score), index=name).sort_values('score', ascending=False)
        # use_feat_numの指定がなければimportance=0のものだけ除去する
        if use_feat_num is None:
            use_feat = feat_df[feat_df['score'] > 0].index.tolist()
        else:
            use_feat = feat_df.index.tolist()[:use_feat_num]

        use_feats.append(use_feat)
    return use_feats


def xgb_feature_select(models_all: List[xgb.Booster], use_feat_num: Optional[int]):
    use_feats = []
    for model in models_all:
        feat_imp = model.get_fscore()
        # use_feat_numの指定がなければimportance=0のものだけ除去する
        if use_feat_num is None:
            use_feat = [col for col, score in model.get_fscore().items() if score > 0]
        else:
            use_feat = [col for col, _ in sorted(model.get_fscore().items(), key=lambda x: x[1], reverse=True)][:use_feat_num]

        use_feats.append(use_feat)
    return use_feats


def select_feature(model_path: Path, model_type: str, use_feat_nums: List[int]):
    """
    feature_importanceによる特徴量選択を行う
    """
    selected_feat_dict = {}
    for level_group in range(3):
        use_feat_num = use_feat_nums[level_group]
        models = pickle.load(open(model_path / f'{model_type}_models_lg{level_group}_non_fs.pkl', 'rb'))
        if model_type == 'lgb':
            selected_feat = lgb_feature_select(models, use_feat_num=use_feat_num)
        elif model_type == 'xgb':
            selected_feat = xgb_feature_select(models, use_feat_num=use_feat_num)
        selected_feat_dict[level_group] = selected_feat
    
    return selected_feat_dict