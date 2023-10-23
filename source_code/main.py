import pandas as pd

# Przykładowy DataFrame
data = {'Kolumna1': ['A', 'B', 'C', 'A', 'B']}
df = pd.DataFrame(data)

# Funkcja do mapowania wartości unikalnych na wartości całkowitoliczbowe
def mapuj_stringi_na_int(df, column_name):
    # Utwórz unikalną listę stringów
    unikalne_stringi = df[column_name].unique()

    # Stwórz słownik przekształcający stringi na inty
    mapa_stringi_na_int = {string: numer for numer, string in enumerate(unikalne_stringi)}

    # Stwórz odwrotny słownik inty na stringi
    mapa_int_na_stringi = {numer: string for string, numer in mapa_stringi_na_int.items()}

    # Dodaj nową kolumnę z przekształconymi wartościami
    df['Nowa_Kolumna'] = df[column_name].map(mapa_stringi_na_int)

    return df, mapa_int_na_stringi

df, mapa_int_na_stringi = mapuj_stringi_na_int(df, 'Kolumna1')

print(df)
print(mapa_int_na_stringi)
