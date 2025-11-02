import pandas as pd

def atribuir_ids_jogadas(df, team_col='possession_team', period_col='period', time_col='timestamp'):
    """
    Adiciona uma coluna 'play_code' ao DataFrame, que identifica cada jogada com base na posse de bola e no período.
    O ID da jogada tem o formato: <equipa>_<periodo>_<número da jogada dessa equipa nesse período>
    """
    
    # Garantir ordenação temporal
    df = df.sort_values(by=[period_col, time_col]).reset_index(drop=True)
    
    # Identificar mudança de posse ou de período
    df['change'] = (df[team_col] != df[team_col].shift()) | (df[period_col] != df[period_col].shift())
    
    # Número sequencial de jogada global (não final)
    df['play_number'] = df['change'].cumsum()
    
    # Para cada equipa e período, numerar sequencialmente as jogadas
    df['play_sequence'] = df.groupby([team_col, period_col])['play_number'].transform(lambda x: pd.factorize(x)[0] + 1)
    
    # Gerar o ID final da jogada
    df['play_code'] = df.apply(
        lambda row: f"{row[team_col]}_{row[period_col]}_{row['play_sequence']}", axis=1
    )
    
    # Limpar colunas auxiliares
    df.drop(columns=['change', 'play_number', 'play_sequence'], inplace=True)
    
    return df
