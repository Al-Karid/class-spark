from pyspark.sql import functions as f
from pyspark.ml.feature import StringIndexer, VectorAssembler, VectorIndexer

def input_df(df):
    ds = df.select('ID_CLIENT',
    f.when(df.LBL_GEO_TRAIN.isin(['Toulouse', 'Lille', 'Dijon',
                                  'Lyon', 'Marseille', 'Paris',
                                  'Nice', 'Limoges','Rouen','Rennes',
                                  'Montpellier', 'Bordeaux', 'Metz',
                                  'Strasbourg']), df.LBL_GEO_TRAIN)\
               .otherwise('na').alias('geo_train'),
    f.when(df.LBL_GEO_AIR.isin(['Aéroports de Paris Orly',
                                'Aéroport de Bâle-Mulhouse / Bassel',
                                'Aéroport Lille Lesquin', 'Aéroport de Rennes',
                                'Aéroport de Nantes Atlantique',
                                'Aéroport de Marseille Provence  (MRS)', 
                                'Aéroport de Bordeaux Mérignac',
                                'Aéroports de Paris Roissy-Charles-de Gaulle', 
                                "Aéroport de Nice Côte d'Azur",
                                'Aéroport de Strasbourg',
                                'Aéroport de Lyon - Saint Exupéry', 
                                'Aéroport de Toulouse Blagnac']), df.LBL_GEO_AIR)\
               .otherwise('na').alias('geo_air'),
    f.when(df.FLG_CMD_CARTE_1225 == '1', '1')\
                   .otherwise('0').alias('cc_jeunes'),
    f.when(df.LBL_STATUT_CLT.isin(['Tres grand', 'Nouveau actif',
                                   'Moyen moins', ' Prospect', ' Petit',
                                   'Inactif', 'Tres petit',
                                   'Nouveau prospect', 'Moyen plus',
                                   'Grand']), df.LBL_STATUT_CLT)\
                   .otherwise('na').alias('segt_rfm'),
    f.when(df.LBL_SEGMENT_ANTICIPATION.isin(['Peu Anticipateur', 'Tres Anticipateur',
                                             'Anticipateur', 'Mixte', 'Non Anticipateur',
                                             'Non Defini']), df.LBL_SEGMENT_ANTICIPATION)\
                   .otherwise('na').alias('segt_anticipation'),
    f.when(df.LBL_SEG_COMPORTEMENTAL.isin(['Mono-commande',
                                           'Comportement Pro',
                                           'Exclusifs Agence', 
                                           'Anticipateurs Methodiques',
                                           'Chasseurs Bons Plans', 
                                           'Rythmes scolaires', 'Nouveaux',
                                           'Sans contraintes']),
           df.LBL_SEG_COMPORTEMENTAL).otherwise('na').alias('segt_comportemental'), 
    f.when(df.LBL_GRP_SEGMENT_NL.isin(['Endormi', 'Spectateur', 'Acteur',
                                       'Eteint', 'Non defini']),
           df.LBL_GRP_SEGMENT_NL).otherwise('na').alias('segt_nl'),
    f.when(((df.AGE > 0) & (df.AGE < 100)), df.AGE)\
                   .otherwise(-1).alias('age'),
    f.when(df.recence_cmd >= 0, df.recence_cmd)\
                   .otherwise(-1).alias('recence_cmd'),
    f.when(((df.mean_duree_voyage > 0) & (df.mean_duree_voyage < 750)),
           df.mean_duree_voyage).otherwise(-1).alias('mean_duree_voyage'),
    f.when(df.days_since_last_visit >= 0, df.days_since_last_visit)\
                   .otherwise(-1).alias('recence_visite'),
    f.when(df.mean_mt_voyage > 0, df.mean_mt_voyage)\
                   .otherwise(-1).alias('mean_mt_voyage'),
    f.when(df.anciennete >= 0, df.anciennete)\
                   .otherwise(-1).alias('anciennete'),
    f.when(df.nb_od > 0, df.nb_od)\
                   .otherwise(-1).alias('nb_od'),
    f.when(df.mean_nb_passagers > 0, df.mean_nb_passagers)\
                   .otherwise(-1).alias('mean_nb_passagers'),
    f.when(df.mean_tarif_loisir >= 0, df.mean_tarif_loisir)\
                   .otherwise(-1).alias('mean_tarif_loisir'),
    f.when(df.mean_classe_1 >= 0, df.mean_classe_1)\
                   .otherwise(-1).alias('mean_classe_1'),
    f.when(df.mean_pointe >= 0, df.mean_pointe)\
                   .otherwise(-1).alias('mean_pointe'),
    f.when(df.mean_depart_we >= 0, df.mean_depart_we)\
                   .otherwise(-1).alias('mean_depart_we'),
    f.when(df.tx_conversion >= 0, df.tx_conversion)\
                   .otherwise(-1).alias('tx_conversion'),
    f.when(df.flg_cmd_lowcost == 1, '1')\
                   .otherwise('0').alias('flg_cmd_lowcost'),
    f.when(df.flg_track_nl_lowcost == 1, '1')\
                   .otherwise('0').alias('flg_track_nl_lowcost'), 
    f.when(df.flg_track_nl == 1, '1')\
                   .otherwise('0').alias('flg_track_nl'))
    
    return ds

def preprocessed_df(df, label="flg_cmd_lowcostIndex"):
    max_values_to_define_str_cols = 10
    id_col = 'ID_CLIENT'
    
    dty = dict(df.dtypes)
    str_cols = [k for k, v in dty.items() if v == 'string']
    str_cols.remove(id_col)
    
    for c in str_cols:
        stringIndexer = StringIndexer(inputCol=c, outputCol=c+"Index")
        model_str = stringIndexer.fit(df)
        df = model_str.transform(df).drop(c)

    input_cols = df.columns
    input_cols.remove(id_col)
    input_cols.remove(label)
    
    assembler = VectorAssembler(inputCols=input_cols,
                            outputCol="features")
    df = assembler.transform(df)
    
    featureIndexer = VectorIndexer(inputCol="features", 
                   outputCol="indexedFeatures", 
                   maxCategories=max_values_to_define_str_cols).fit(df)
    return featureIndexer.transform(df), df