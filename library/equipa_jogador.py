# equipa_jogador.py

def atribuir_equipa_jogador(df, equipa_esquerda, equipa_oponente):
    def obter(row):
        if row["teammate"]:
            return row["equipa_actor_frame"]
        else:
            return equipa_oponente if row["equipa_actor_frame"] == equipa_esquerda else equipa_esquerda

    df["equipa_jogador"] = df.apply(obter, axis=1)
    return df
