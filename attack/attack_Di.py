#使用机器学习模型 DDi.json 进行成员推断攻击
#这些攻击主要是利用训练好的模型（比如 XGBoost），通过模型的行为推断哪些样本（Ai.csv）在训练集(Bi.csv)里。
from abc import ABC
import sys, os
import argparse

import xgboost as xgb
import pandas as pd


# モジュールの相対参照制限を強制的に回避
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'analysis'))
from xgbt_train import build_X

TARGET = "stroke_flag"   #目标列名
NUM_FEATURES = 21
NUM_CLASSES = 2

#抽象基类：加载模型、对齐特征
class Attack_Di_Base(ABC):
    def __init__(self, path_to_xgbt_model_json):
        """
        攻撃者の初期化

        path_to_xgbt_model_json: 学習済みのxgboostモデルのjsonファイルへのパス
        """
        # json fileを読み込み
        xgbt_model = xgb.Booster()
        xgbt_model.load_model(path_to_xgbt_model_json)

        self.xgbt_model = xgbt_model

        #特征矩阵X，标签Y，推断结果inferred
        self.X = None
        self.y = None
        self.inferred = None

    #准备工作：特征对齐+取标签；不判定
    def infer(self, path_to_Ai_csv):
        Ai_df = pd.read_csv(path_to_Ai_csv, dtype=str, keep_default_na=False)
        
        # 説明変数と目的変数に分割
        X = build_X(Ai_df, TARGET)

        # Xのみにある列は削除する, 9/10追記
        #删除仅存在与当前X的但不在模型feature_names中的列
        columns_only_X = set(X.columns) - set(self.xgbt_model.feature_names)
        if columns_only_X:
            X = X.drop(columns=columns_only_X)

        # xgbt_model.feature_namesのみにある列は0埋め, 9/10追記
        #对于模型存在，Ai缺失的列，用0填充
        columns_only_feature_names = set(self.xgbt_model.feature_names) - set(X.columns)
        if columns_only_feature_names:
            for col in columns_only_feature_names:
                # 0で埋める
                X[col] = 0

        # Xの列をXGBoostモデルが要求する順番に並び替え, 9/10追記
        X = X.reindex(columns=self.xgbt_model.feature_names)

        X.columns = self.xgbt_model.feature_names
        self.X = X.copy()
        self.y = pd.to_numeric(Ai_df[TARGET], errors="coerce").astype(int).values

        # print(set(self.xgbt_model.feature_names)-set(X.columns.tolist()))

        return None
    
    def save_inferred(self, path_to_output):
        if self.inferred is None:
            print("inferred is None. No file was saved.")
        else:
            self.inferred.to_csv(path_to_output, index=False, header=False)
            print("inferred was successfully saved.")

#如果模型预测正确，就认为这行是“成员”；预测错了，就认为是“非成员”。
class Pred_Attack(Attack_Di_Base):
    """
    モデルが正答した行をmemberと推定する
    """
    def __init__(self, path_to_xgboost_model_json):
        super().__init__(path_to_xgboost_model_json)

    def infer(self, path_to_Ai_csv):
        super().infer(path_to_Ai_csv)

        pred = self.xgbt_model.predict(xgb.DMatrix(self.X))  #用Booster.predict(DMatrix(X)) 得到正类概率
        pred[pred<0.5] = 0
        pred[pred>=0.5] = 1
        
        inferred = pd.DataFrame(pred == self.y, dtype=int)   #判定
        self.inferred = inferred

        return inferred

#不仅看预测对不对，还看模型的“信心”。如果模型非常有把握（超过某个阈值），就判断为“成员”。阈值可以自己设定。
class Conf_Attack(Attack_Di_Base):
    """
    モデルが確信を持って正答した行をmemberと推定する
    """
    def __init__(self, path_to_xgboost_model_json, threshold=0.1):
        super().__init__(path_to_xgboost_model_json)
        self.threshold = threshold

    def infer(self, path_to_Ai_csv):
        super().infer(path_to_Ai_csv)

        pred = self.xgbt_model.predict(xgb.DMatrix(self.X))
        confidence = pd.DataFrame(pred-self.y).abs()     #计算 |pred − y|≤ 阈值 ⇒ 记 1（成员）；默认阈值 0.1（即 p ≥ 0.9 / p ≤ 0.1）。
        inferred = (confidence <= self.threshold)
        self.inferred = inferred

        return inferred

#选最有把握的前 10000 行
class TopConfAttack(Attack_Di_Base):
    """
    モデルの確信度top1000の行をmemberと推定
    """
    def infer(self, path_to_Ai_csv):
        super().infer(path_to_Ai_csv)

        pred = self.xgbt_model.predict(xgb.DMatrix(self.X))
        confidence = (-1) * pd.DataFrame(pred-self.y, columns=["conf"]).abs()   # (-1)*|pred − y|
        inferred = pd.DataFrame(0, index=range(pred.shape[0]), columns=["inferred"])
        inferred.loc[confidence["conf"].nlargest(10000).index, "inferred"] = 1  #选 Top 10000 个置信度最高的样本记为 1，其余 0。
        self.inferred = inferred

        return inferred

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="")
    ap.add_argument("model_json", help="trained model JSON (Booster.save_model)")
    ap.add_argument("Ai_csv", help="Ai.csv to attack")
    args = ap.parse_args()

    #正确性
    attacker = Pred_Attack(args.model_json)
    pred = attacker.infer(args.Ai_csv)
    attacker.save_inferred("inferred_membership1.csv")

    #置信度阈值
    attacker = Conf_Attack(args.model_json)
    pred = attacker.infer(args.Ai_csv)
    attacker.save_inferred("inferred_membership2.csv")

    #TOP 10000
    attacker = TopConfAttack(args.model_json)
    pred = attacker.infer(args.Ai_csv)
    attacker.save_inferred("inferred_membership3.csv")
