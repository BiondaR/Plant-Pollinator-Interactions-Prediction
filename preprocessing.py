#######################
# Trabalho 2          #
# Grafos e Aplicações #
# Bionda Rozin        #
#######################

# Imports
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import utils as u

def preprocessing():
    df = pd.read_csv("Data\\plant_pollinator_database.csv")
    df.fillna(0, inplace=True)

    plants_cols = ["crop", "type", "season", "diameter", "corolla", "colour", "nectar", "b.system", "s.pollination", "inflorescence", "composite"]
    pol_cols = ["visitor", "guild", "tongue", "body", "sociality", "feeding"]

    plants_df = u.select_columns(df, plants_cols, "crop").to_dict(orient="records")
    pol_df = u.select_columns(df, pol_cols, "visitor").to_dict(orient="records")

    vec = DictVectorizer()
    plants_feats = vec.fit_transform(plants_df).toarray()
    pol_feats = vec.fit_transform(pol_df).toarray()

    # montar uma matriz em que todos interagem com todos
    features = []
    for i in plants_feats:
        for j in pol_feats:
            features.append(i.tolist()+j.tolist())

    # ver no dataframe quais elementos têm interação (1). Se não achar, definir como 0
    plants = df["crop"].drop_duplicates()
    pollinators = df["visitor"].drop_duplicates()

    labels = []
    for i in plants:
        for j in pollinators:
            if ((df["crop"] == i) & (df["visitor"] == j)).any():
                labels.append(1)
            else:
                labels.append(0)

    return features, labels