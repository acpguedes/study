# %%
import sys
import pandas as pd
import numpy as np
from typing import List, Dict
from faker import Faker
import random
import datetime

def create_dataframe(size: int = 10)-> pd.DataFrame:
    """
    Cria um dataframe com dados fakes de acordo com o tamanho passado como argumento.
    :param size: int: Tamanho do dataframe a ser criado (padrão é 10)
    :return: pd.DataFrame: Dataframe com dados fakes
    """

    fake = Faker()

    # Cria um dataframe com dados de 10 pessoas
    df = pd.DataFrame({
        "cliente": [fake.name() for _ in range(size)],
        "age": [fake.random_int(min=18, max=80, step=1) for _ in range(size)],
        "location": [random.choice(["interior", "cidade", "rural"]) for _ in range(size)]
    })
    return df


def gerar_faturas(
        clientes: List[str],
        n_faturas_min: int = 20,
        n_faturas_max: int = 60,
        valor_minimo: float = 60,
        valor_maximo: float = 500,
        sd_min: float = 2,
        sd_max: float = 1
)-> pd.DataFrame:
    """Gera faturamento de clientes de forma aleatória.
    
    :param clientes: List[str]: Lista com os nomes dos clientes.
    :param n_faturas_min: int: Número mínimo de faturas a serem geradas para cada cliente.
    :param n_faturas_max: int: Número máximo de faturas a serem geradas para cada cliente.
    :param valor_minimo: float: Valor mínimo das faturas.
    :param valor_maximo: float: Valor máximo das faturas.
    :param sd_min: float: Desvio padrão mínimo.
    :param sd_max: float: Desvio padrão máximo.

    :return: pd.DataFrame: Tabela com as faturas geradas.
    """
    dados_faturas = []

    num_faturas_por_cliente = np.random.randint(n_faturas_min, n_faturas_max + 1, size=len(clientes))
    medias_faturas = np.random.uniform(valor_minimo, valor_maximo, size=len(clientes))
    desvios_padrao = np.random.uniform(sd_min, sd_max, size=len(clientes))

    for i, cliente in enumerate(clientes):
        
        valores_faturas = np.random.normal(medias_faturas[i], desvios_padrao[i], num_faturas_por_cliente[i])

        for valor in valores_faturas:
            dados_faturas.append([cliente, valor])

    df_faturas = pd.DataFrame(dados_faturas, columns=['cliente', 'vl_fatura'])

    df_faturas['numero_fatura'] = df_faturas.groupby('cliente').cumcount() + 1

    return df_faturas

def marcar_faturas_pagas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Marca aleatoriamente as faturas como pagas (True) ou não pagas (False), garantindo que a fatura mais recente
    esteja sempre marcada como não paga (False) e as duas últimas faturas de cada cliente possam estar pagas ou não.

    :param df: pd.DataFrame: DataFrame com os dados das faturas dos clientes.
    :return: pd.DataFrame: DataFrame com a coluna 'status_pago' adicionada, indicando se a fatura foi paga (True) ou não (False).
    """

    # Inicializa a coluna 'status_pago' com True (paga)
    df['status_pago'] = True
    
    # Marca a fatura mais recente de cada cliente como False (não paga)
    df.loc[df.groupby('cliente')['numero_fatura'].idxmax(), 'status_pago'] = False
    
    # Para cada cliente, identifica aleatoriamente uma das duas últimas faturas (exceto a mais recente) para ser potencialmente marcada como False (não paga)
    for cliente in df['cliente'].unique():
        faturas_cliente = df[df['cliente'] == cliente]
        max_numero_fatura = faturas_cliente['numero_fatura'].max()
        
        if max_numero_fatura > 2:  # Se existe mais de uma fatura além da mais recente
            fatura_aleatoria = random.choice(range(max_numero_fatura - 1, max_numero_fatura))  # Seleciona entre a penúltima e a antepenúltima
            df.loc[(df['cliente'] == cliente) & (df['numero_fatura'] == fatura_aleatoria), 'status_pago'] = False
    
    return df


def calcular_media_vl_fatura(df: pd.DataFrame, coluna_valor: str, coluna_saida: str) -> pd.DataFrame:
    """
    Calcula a média do valor das 12 faturas anteriores para cada fatura de cada cliente,
    aplicando regras específicas para os primeiros registros.

    :param df: pd.DataFrame: DataFrame com os dados das faturas dos clientes.
    :param coluna_valor: str: Nome da coluna do valor das faturas.
    :param coluna_saida: str: Nome da coluna de saída para as médias calculadas.
    :return: pd.DataFrame: DataFrame com a coluna de saída adicionada.
    """
    # Ordena o DataFrame por cliente e número da fatura para garantir a sequência correta
    df.sort_values(by=['cliente', 'numero_fatura'], inplace=True)

    # Inicializa a coluna de saída com NaN
    df[coluna_saida] = np.nan

    for cliente in df['cliente'].unique():
        cliente_indices = df[df['cliente'] == cliente].index
        for i in cliente_indices:
            numero_fatura = df.at[i, 'numero_fatura']
            if numero_fatura == 1 or numero_fatura == 2:
                df.at[i, coluna_saida] = df.at[cliente_indices[0], coluna_valor]
            else:
                # Calcula a média das 12 faturas anteriores, se disponível
                faturas_anteriores = df.loc[cliente_indices[df.loc[cliente_indices, 'numero_fatura'] < numero_fatura], coluna_valor]
                df.at[i, coluna_saida] = faturas_anteriores.tail(12).mean()

    return df


def calcular_sd_vl_fatura(df: pd.DataFrame, coluna_valor: str, coluna_saida: str) -> pd.DataFrame:
    """
    Calcula o desvio padrão do valor das 12 faturas anteriores para cada fatura de cada cliente,
    aplicando regras específicas para os primeiros registros.

    :param df: pd.DataFrame: DataFrame com os dados das faturas dos clientes.
    :param coluna_valor: str: Nome da coluna do valor das faturas para cálculo do desvio padrão.
    :param coluna_saida: str: Nome da coluna de saída para os desvios padrão calculados.
    :return: pd.DataFrame: DataFrame com a coluna de saída adicionada.
    """
    # Ordena o DataFrame por cliente e número da fatura para garantir a sequência correta
    df.sort_values(by=['cliente', 'numero_fatura'], inplace=True)

    # Inicializa a coluna de saída com 0
    df[coluna_saida] = np.nan

    for cliente in df['cliente'].unique():
        cliente_indices = df[df['cliente'] == cliente].index
        for i in cliente_indices:
            numero_fatura = df.at[i, 'numero_fatura']
            if numero_fatura == 1 or numero_fatura == 2:
                df.at[i, coluna_saida] = 0
            else:
                # Calcula o desvio padrão das 12 faturas anteriores, se disponível
                faturas_anteriores = df.loc[cliente_indices[df.loc[cliente_indices, 'numero_fatura'] < numero_fatura], coluna_valor]
                df.at[i, coluna_saida] = faturas_anteriores.tail(12).std()

    return df

def calcular_zscore_faturas(df: pd.DataFrame, coluna_valor: str, coluna_media: str, coluna_sd: str, coluna_saida: str) -> pd.DataFrame:
    """
    Calcula o z-score de cada fatura com base na média e desvio padrão das 12 faturas anteriores.
    Retorna 0 para o z-score se o desvio padrão for 0.

    :param df: pd.DataFrame: DataFrame com os dados das faturas dos clientes.
    :param coluna_valor: str: Nome da coluna do valor das faturas.
    :param coluna_media: str: Nome da coluna que contém a média das faturas anteriores.
    :param coluna_sd: str: Nome da coluna que contém o desvio padrão das faturas anteriores.
    :param coluna_saida: str: Nome da coluna de saída para os z-scores calculados.
    :return: pd.DataFrame: DataFrame com a coluna de saída adicionada.
    """
    # Calcula o z-score, retornando 0 quando o desvio padrão é 0
    df[coluna_saida] = np.where(df[coluna_sd] != 0,
                                (df[coluna_valor] - df[coluna_media]) / df[coluna_sd],
                                0)
    
    return df

def marcar_pagamento_antes_vencimento(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona uma coluna booleana ao DataFrame indicando se a fatura foi paga antes do vencimento.
    Para faturas onde 'status_pago' é False, o valor da nova coluna também será False.
    Para as demais faturas, o valor será determinado aleatoriamente.

    :param df: pd.DataFrame: DataFrame com os dados das faturas dos clientes, incluindo a coluna 'status_pago'.
    :return: pd.DataFrame: DataFrame com a coluna 'pago_antes_vencimento' adicionada.
    """
    
    # Gera valores aleatórios para 'pago_antes_vencimento', exceto onde 'status_pago' é False
    df['pago_antes_vencimento'] = df['status_pago'].apply(lambda x: random.choice([True, False]) if x else False)
    
    return df

def calcular_dias_pagamento(df: pd.DataFrame, dias_min_atraso: int = 1, dias_max_atraso: int = 100, dias_min_antecipado: int = -5, dias_max_antecipado: int = 0) -> pd.DataFrame:
    """
    Adiciona uma coluna ao DataFrame com o número inteiro de dias de atraso ou antecipação para faturas.
    Para faturas não pagas antes do vencimento, os dias de atraso são gerados aleatoriamente dentro de um
    intervalo definido. Para faturas pagas antes do vencimento, os dias de antecipação são gerados entre -5 e 0.

    :param df: pd.DataFrame: DataFrame com os dados das faturas dos clientes, incluindo a coluna 'pago_antes_vencimento'.
    :param dias_min_atraso: int: Valor mínimo de dias de atraso (padrão é 1).
    :param dias_max_atraso: int: Valor máximo de dias de atraso (padrão é 100).
    :param dias_min_antecipado: int: Valor mínimo de dias de antecipação (padrão é -5).
    :param dias_max_antecipado: int: Valor máximo de dias de antecipação (padrão é 0).
    :return: pd.DataFrame: DataFrame com a coluna 'dias_pagamento' adicionada.
    """
    
    # Define a função para calcular os dias de pagamento baseado no status de pagamento
    def calcular_dias(x):
        if x['pago_antes_vencimento']:
            return random.randint(dias_min_antecipado, dias_max_antecipado)
        else:
            return random.randint(dias_min_atraso, dias_max_atraso)

    # Aplica a função acima para cada linha do DataFrame e cria a coluna 'dias_pagamento'
    df['dias_pagamento'] = df.apply(calcular_dias, axis=1)
    
    return df

def calcular_frequencia_faturas_aberto_12_meses(
        df: pd.DataFrame, coluna_status: str, coluna_saida: str
) -> pd.DataFrame:
    """
    Calcula a frequência de faturas em aberto nos 12 meses anteriores para cada fatura de cada cliente.

    :param df: pd.DataFrame: DataFrame com os dados das faturas dos clientes.
    :param coluna_status: str: Nome da coluna que indica se a fatura está em aberto.
    :param coluna_saida: str: Nome da coluna de saída para a frequência de faturas em aberto nos 12 meses anteriores.
    :return: pd.DataFrame: DataFrame com a coluna de saída adicionada.
    """
    # Inicializa a coluna de saída com 0
    df[coluna_saida] = 0

    # Ordena o DataFrame por cliente e número da fatura para garantir a sequência correta
    df.sort_values(by=['cliente', 'numero_fatura'], inplace=True)

    for cliente in df['cliente'].unique():
        faturas_cliente = df[df['cliente'] == cliente]
        for idx, fatura in faturas_cliente.iterrows():
            # Identifica o intervalo de 12 meses anteriores para a fatura atual
            limite_superior = fatura['numero_fatura'] - 1
            limite_inferior = max(1, limite_superior - 12)
            # Seleciona as faturas dentro desse intervalo
            faturas_anteriores = faturas_cliente[(faturas_cliente['numero_fatura'] <= limite_superior) & (faturas_cliente['numero_fatura'] > limite_inferior)]
            # Calcula a quantidade de faturas em aberto nesse intervalo
            df.at[idx, coluna_saida] = sum(faturas_anteriores[coluna_status].apply(lambda x: 1 if not x else 0))

    return df

def calcular_total_devido(df: pd.DataFrame, coluna_status_pago: str, coluna_pago_antes_vencimento: str, coluna_valor: str, coluna_saida: str) -> pd.DataFrame:
    """
    Calcula o total devido somando as faturas não pagas nos últimos 12 meses, ou nos últimos 3 meses
    se a fatura atual foi paga a tempo, mas apenas para faturas não pagas antes do vencimento.

    :param df: pd.DataFrame: DataFrame com os dados das faturas dos clientes.
    :param coluna_status_pago: str: Nome da coluna que indica se a fatura foi paga.
    :param coluna_pago_antes_vencimento: str: Nome da coluna que indica se a fatura foi paga antes do vencimento.
    :param coluna_valor: str: Nome da coluna do valor das faturas.
    :param coluna_saida: str: Nome da coluna de saída para o total devido.
    :return: pd.DataFrame: DataFrame com a coluna de saída adicionada.
    """
    # Inicializa a coluna de saída com 0
    df[coluna_saida] = 0.0

    # Ordena o DataFrame por cliente e número da fatura para garantir a sequência correta
    df.sort_values(by=['cliente', 'numero_fatura'], inplace=True)

    for cliente in df['cliente'].unique():
        faturas_cliente = df[df['cliente'] == cliente]
        for idx, fatura_atual in faturas_cliente.iterrows():
            numero_fatura_atual = fatura_atual['numero_fatura']
            if fatura_atual[coluna_status_pago]:
                # Considera as faturas dos últimos 3 meses que não foram pagas antes do vencimento
                limite_inferior = max(1, numero_fatura_atual - 3)
                faturas_analisadas = faturas_cliente[(faturas_cliente['numero_fatura'] >= limite_inferior) & 
                                                     (faturas_cliente['numero_fatura'] < numero_fatura_atual) &
                                                     (faturas_cliente[coluna_pago_antes_vencimento] == False)]
            else:
                # Considera todas as faturas não pagas nos últimos 12 meses
                limite_inferior = max(1, numero_fatura_atual - 12)
                faturas_analisadas = faturas_cliente[(faturas_cliente['numero_fatura'] >= limite_inferior) & 
                                                     (faturas_cliente['numero_fatura'] < numero_fatura_atual) & 
                                                     (faturas_cliente[coluna_status_pago] == False)]
            
            total_devido = faturas_analisadas[coluna_valor].sum()
            df.at[idx, coluna_saida] = total_devido

    return df


def build_fake_dataframe(size: int, 
                            n_faturas_min: int, n_faturas_max: int, 
                            valor_minimo: float, valor_maximo: float, 
                            sd_min: float, sd_max: float, 
                            coluna_cliente: str, 
                            coluna_vl_fatura: str, 
                            coluna_status_pago: str, 
                            coluna_pago_antes_vencimento: str, 
                            coluna_numero_fatura: str,
                            dias_min_atraso: int, dias_max_atraso: int, 
                            dias_min_antecipado: int, dias_max_antecipado: int,
                            coluna_dias_pagamento: str,
                            coluna_media_vl_fatura: str, 
                            coluna_sd_vl_fatura: str, 
                            coluna_zscore_faturas: str,
                            coluna_media_dias_pagamento: str, 
                            coluna_sd_dias_pagamento: str, 
                            coluna_zscore_dias_pagamento: str,
                            coluna_frequencia_faturas_aberto_12_meses: str, 
                            coluna_total_devido: str,
                            coluna_media_total_devido: str, 
                            coluna_sd_total_devido: str, 
                            coluna_zscore_total_devido: str) -> pd.DataFrame:
    """
    Gera um dataframe com dados fictícios de clientes e faturas.

    :param size: Número de clientes a serem gerados.
    :type size: int
    :param n_faturas_min: Número mínimo de faturas a serem geradas para cada cliente.
    :type n_faturas_min: int
    :param n_faturas_max: Número máximo de faturas a serem geradas para cada cliente.
        Se for None, assume o mesmo número que n_faturas_min.
    :type n_faturas_max: int or None
    :param valor_minimo: Valor mínimo das faturas.
    :type valor_minimo: float
    :param valor_maximo: Valor máximo das faturas.
    :type valor_maximo: float
    :param sd_min: Desvio padrão mínimo.
    :type sd_min: float
    :param sd_max: Desvio padrão máximo.
    :type sd_max: float
    :param coluna_cliente: Nome da coluna que contém o nome do cliente.
    :type coluna_cliente: str
    :param coluna_vl_fatura: Nome da coluna que contém o valor da fatura.
    :type coluna_vl_fatura: str
    :param coluna_status_pago: Nome da coluna que indica se a fatura foi paga.
    :type coluna_status_pago: str
    :param coluna_pago_antes_vencimento: Nome da coluna que contém a data de vencimento da fatura.
    :type coluna_pago_antes_vencimento: str
    :param coluna_numero_fatura: Nome da coluna que contém o número da fatura.
    :type coluna_numero_fatura: str
    :param dias_min_atraso: Valor mínimo de dias de atraso.
    :type dias_min_atraso: int
    :param dias_max_atraso: Valor máximo de dias de atraso.
    :type dias_max_atraso: int
    :param dias_min_antecipado: Valor mínimo de dias de antecipação.
    :type dias_min_antecipado: int
    :param dias_max_antes_vencimento: Valor máximo de dias de antecipação.
    :type dias_max_antes_vencimento: int
    :type coluna_dias_pagamento: str, optional
    :param media_total_devido: Nome da coluna de saída para o total devido.
    :type coluna_media_vl_fatura: str, optional
    : param coluna_media_vl_fatura: Nome da coluna de saída para as médias calculadas.
    :type coluna_sd_vl_fatura: str, optional
    :param coluna_sd_vl_fatura: Nome da coluna de saída para os desvios padrão calculados.
    :type coluna_zscore_faturas: str, optional
    :param coluna_zscore_faturas: Nome da coluna de saída para os z-scores calculados.
    :type coluna_media_dias_pagamento: str, optional
    :param coluna_media_dias_pagamento: Nome da coluna de saída para as médias calculadas.
    :type coluna_sd_dias_pagamento: str, optional
    :param coluna_sd_dias_pagamento: Nome da coluna de saída para os desvios padrão calculados.
    :type coluna_zscore_dias_pagamento: str, optional
    :param coluna_zscore_dias_pagamento: Nome da coluna de saída para os z-scores calculados.
    :type coluna_frequencia_faturas_aberto_12_meses: str, optional
    :param coluna_frequencia_faturas_aberto_12_meses: Nome da coluna de saída para a frequência de faturas em aberto nos 12 meses anteriores.
    :type coluna_total_devido: str, optional
    :param coluna_total_devido: Nome da coluna de saída para o total devido.
    :type coluna_media_total_devido: str, optional
    :param coluna_media_total_devido: Nome da coluna de saída para as médias calculadas.
    :type coluna_sd_total_devido: str, optional
    :param coluna_sd_total_devido: Nome da coluna de saída para os desvios padrão calculados.
    :type coluna_zscore_total_devido: str, optional
    :param coluna_zscore_total_devido: Nome da coluna de saída para os z-scores calculados.
    
    return: DataFrame com os dados fictícios de clientes e faturas.
    :rtype: pandas.core.frame.DataFrame
    """
    df_clientes = create_dataframe(size=size)
    df_faturas = gerar_faturas(df_clientes[coluna_cliente].unique(), n_faturas_min, n_faturas_max, 
                               valor_minimo, valor_maximo, sd_min, sd_max)
    df = df_clientes.merge(df_faturas, on=coluna_cliente)
    df = (df
          .pipe(marcar_faturas_pagas)
          .pipe(calcular_media_vl_fatura, coluna_vl_fatura, coluna_media_vl_fatura)
          .pipe(calcular_sd_vl_fatura, coluna_vl_fatura, coluna_sd_vl_fatura)
          .pipe(calcular_zscore_faturas, coluna_vl_fatura, coluna_media_vl_fatura, coluna_sd_vl_fatura, coluna_zscore_faturas)
          .pipe(marcar_pagamento_antes_vencimento)
          .pipe(calcular_dias_pagamento, dias_min_atraso, dias_max_atraso, dias_min_antecipado, dias_max_antecipado)
          .pipe(calcular_media_vl_fatura, coluna_dias_pagamento, coluna_media_dias_pagamento)
          .pipe(calcular_sd_vl_fatura, coluna_dias_pagamento, coluna_sd_dias_pagamento)
          .pipe(calcular_zscore_faturas, coluna_dias_pagamento, coluna_media_dias_pagamento, coluna_sd_dias_pagamento, coluna_zscore_dias_pagamento)
          .pipe(calcular_frequencia_faturas_aberto_12_meses, coluna_pago_antes_vencimento, coluna_frequencia_faturas_aberto_12_meses)
          .pipe(calcular_total_devido, coluna_status_pago, coluna_pago_antes_vencimento, coluna_vl_fatura, coluna_total_devido)
          .pipe(calcular_media_vl_fatura, coluna_total_devido, coluna_media_total_devido)
          .pipe(calcular_sd_vl_fatura, coluna_total_devido, coluna_sd_total_devido)
          .pipe(calcular_zscore_faturas, coluna_total_devido, coluna_media_total_devido, coluna_sd_total_devido, coluna_zscore_total_devido)
         )
    return df


def generate_fake_dataframe(size: int = 100) -> pd.DataFrame:
    """
    Executa a geração de um dataframe com dados fictícios de clientes e faturas utilizando os valores padrão
    e imprime o resultado no formato CSV.
    """
    df = build_fake_dataframe(
        size=size, n_faturas_min=20, n_faturas_max=60,
        valor_minimo=60, valor_maximo=500, 
        sd_min=2, sd_max=5,
        coluna_cliente='cliente', coluna_vl_fatura='vl_fatura', 
        coluna_status_pago='status_pago', 
        coluna_pago_antes_vencimento='pago_antes_vencimento', 
        coluna_numero_fatura='numero_fatura',
        dias_min_atraso=1, dias_max_atraso=100, 
        dias_min_antecipado=-5, dias_max_antecipado=0,
        coluna_dias_pagamento='dias_pagamento',
        coluna_media_vl_fatura='media_vl_fatura', 
        coluna_sd_vl_fatura='sd_vl_fatura', 
        coluna_zscore_faturas='zscore_faturas',
        coluna_media_dias_pagamento='media_dias_pagamento', 
        coluna_sd_dias_pagamento='sd_dias_pagamento', 
        coluna_zscore_dias_pagamento='zscore_dias_pagamento',
        coluna_frequencia_faturas_aberto_12_meses='frequencia_faturas_aberto_12_meses', 
        coluna_total_devido='total_devido',
        coluna_media_total_devido='media_total_devido', 
        coluna_sd_total_devido='sd_total_devido', 
        coluna_zscore_total_devido='zscore_total_devido'
    )
    return df

def main(size: int = 100) -> None:
    """
    Função principal que executa a geração de um dataframe com dados fictícios de clientes e faturas utilizando os valores padrão.
    """
    df_resultado = generate_fake_dataframe(size)
    df_resultado.to_csv(sys.stdout, index=False)

if __name__ == "__main__":
    # Verifica se um argumento de tamanho foi fornecido na linha de comando.
    if len(sys.argv) > 1:
        try:
            size = int(sys.argv[1])  # Tenta converter o segundo argumento para inteiro.
        except ValueError:
            print("O argumento fornecido não é um inteiro válido. Usando o valor padrão de 100.")
            size = 100
    else:
        size = 100  # Usa o valor padrão se nenhum argumento for fornecido.

    main(size)
