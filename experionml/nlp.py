from __future__ import annotations

import re
import unicodedata
from string import punctuation
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from beartype import beartype
from sklearn.base import OneToOneFeatureMixin
from sklearn.utils.validation import _check_feature_names_in
from typing_extensions import Self

from experionml.data_cleaning import TransformerMixin
from experionml.utils.types import (
    Bool,
    Engine,
    FloatLargerZero,
    Sequence,
    VectorizerStarts,
    Verbose,
    XConstructor,
    XReturn,
    YConstructor,
    bool_t,
)
from experionml.utils.utils import (
    check_is_fitted,
    check_nltk_module,
    get_corpus,
    is_sparse,
    merge,
    to_df,
)


if TYPE_CHECKING:
    from nltk.corpus import wordnet


@beartype
class TextCleaner(TransformerMixin, OneToOneFeatureMixin):
    r"""Aplica limpeza de texto padrão ao corpus.

    As transformações incluem normalização de caracteres e remoção
    de ruídos do texto (e-mails, tags HTML, URLs, etc...). As
    transformações são aplicadas na coluna chamada `corpus`, na
    mesma ordem em que os parâmetros são apresentados. Se não houver
    uma coluna com esse nome, uma exceção é lançada.

    Esta classe pode ser acessada no experionml através do método [textclean]
    [experionmlclassifier-textclean]. Leia mais no [guia do usuário]
    [text-cleaning].

    Parâmetros
    ----------
    decode: bool, default=True
        Se deve decodificar caracteres unicode para suas representações ascii.

    lower_case: bool, default=True
        Se deve converter todos os caracteres para minúsculas.

    drop_email: bool, default=True
        Se deve remover endereços de e-mail do texto.

    regex_email: str, default=None
        Regex usado para buscar endereços de e-mail. Se None, usa
        `r"[\w.-]+@[\w-]+\.[\w.-]+"`.

    drop_url: bool, default=True
        Se deve remover links de URL do texto.

    regex_url: str, default=None
        Regex usado para buscar URLs. Se None, usa
        `r"https?://\S+|www\.\S+"`.

    drop_html: bool, default=True
        Se deve remover tags HTML do texto. Esta opção é
        particularmente útil se os dados foram extraídos de um site.

    regex_html: str, default=None
        Regex usado para buscar tags HTML. Se None, usa
        `r"<.*?>"`.

    drop_emoji: bool, default=True
        Se deve remover emojis do texto.

    regex_emoji: str, default=None
        Regex usado para buscar emojis. Se None, usa
        `r":[a-z_]+:"`.

    drop_number: bool, default=True
        Se deve remover números do texto.

    regex_number: str, default=None
        Regex usado para buscar números. Se None, usa
        `r"\b\d+\b".` Note que números adjacentes a letras não
        são removidos.

    drop_punctuation: bool, default=True
        Se deve remover pontuações do texto. Os caracteres
        considerados pontuação são `!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`.

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não imprimir nada.
        - 1 para imprimir informações básicas.
        - 2 para imprimir informações detalhadas.

    Veja também
    --------
    experionml.nlp:TextNormalizer
    experionml.nlp:Tokenizer
    experionml.nlp:Vectorizer

    Exemplos
    --------
    === "experionml"
        ```pycon
        import numpy as np
        from experionml import ExperionMLClassifier
        from sklearn.datasets import fetch_20newsgroups

        X, y = fetch_20newsgroups(
            return_X_y=True,
            categories=["alt.atheism", "sci.med", "comp.windows.x"],
            shuffle=True,
            random_state=1,
        )
        X = np.array(X).reshape(-1, 1)

        experionml = ExperionMLClassifier(X, y, random_state=1)
        print(experionml.dataset)

        experionml.textclean(verbose=2)

        print(experionml.dataset)
        ```

    === "stand-alone"
        ```pycon
        import numpy as np
        from experionml.nlp import TextCleaner
        from sklearn.datasets import fetch_20newsgroups

        X, y = fetch_20newsgroups(
            return_X_y=True,
            categories=["alt.atheism", "sci.med", "comp.windows.x"],
            shuffle=True,
            random_state=1,
        )
        X = np.array(X).reshape(-1, 1)

        textcleaner = TextCleaner(verbose=2)
        X = textcleaner.transform(X)

        print(X)
        ```

    """

    def __init__(
        self,
        *,
        decode: Bool = True,
        lower_case: Bool = True,
        drop_email: Bool = True,
        regex_email: str | None = None,
        drop_url: Bool = True,
        regex_url: str | None = None,
        drop_html: Bool = True,
        regex_html: str | None = None,
        drop_emoji: Bool = True,
        regex_emoji: str | None = None,
        drop_number: Bool = True,
        regex_number: str | None = None,
        drop_punctuation: Bool = True,
        verbose: Verbose = 0,
    ):
        super().__init__(verbose=verbose)
        self.decode = decode
        self.lower_case = lower_case
        self.drop_email = drop_email
        self.regex_email = regex_email
        self.drop_url = drop_url
        self.regex_url = regex_url
        self.drop_html = drop_html
        self.regex_html = regex_html
        self.drop_emoji = drop_emoji
        self.regex_emoji = regex_emoji
        self.drop_number = drop_number
        self.regex_number = regex_number
        self.drop_punctuation = drop_punctuation

    def transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Aplica as transformações aos dados.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de features com shape=(n_samples, n_features). Se X não
            for um dataframe, deve ser composto por uma única feature
            contendo os documentos de texto.

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        Retorna
        -------
        dataframe
            Corpus transformado.

        """

        def to_ascii(elem: str) -> str:
            """Converte string unicode para ascii.

            Parâmetros
            ----------
            elem: str
                Elementos do corpus.

            Retorna
            -------
            str
                String ASCII.

            """
            try:
                elem.encode("ASCII", errors="strict")  # Returns byes object
            except UnicodeEncodeError:
                norm = unicodedata.normalize("NFKD", elem)
                return "".join([c for c in norm if not unicodedata.combining(c)])
            else:
                return elem  # Retorna sem alterações se a codificação foi bem-sucedida

        def drop_regex(regex: str):
            """Encontra e remove uma expressão regex do corpus.

            Parâmetros
            ----------
            regex: str
                Padrão regex a ser substituído.

            """
            if isinstance(Xt[corpus].iloc[0], str):
                Xt[corpus] = Xt[corpus].str.replace(regex, "", regex=True)
            else:
                Xt[corpus] = Xt[corpus].apply(lambda x: [re.sub(regex, "", w) for w in x])

        Xt = to_df(X, columns=getattr(self, "feature_names_in_", None))
        corpus = get_corpus(Xt)

        self._log("Limpando o corpus...", 1)

        if self.decode:
            if isinstance(Xt[corpus].iloc[0], str):
                Xt[corpus] = Xt[corpus].apply(lambda x: to_ascii(x))
            else:
                Xt[corpus] = Xt[corpus].apply(lambda doc: [to_ascii(str(w)) for w in doc])
        self._log(" --> Decodificando caracteres unicode para ascii.", 2)

        if self.lower_case:
            self._log(" --> Convertendo texto para minúsculas.", 2)
            if isinstance(Xt[corpus].iloc[0], str):
                Xt[corpus] = Xt[corpus].str.lower()
            else:
                Xt[corpus] = Xt[corpus].apply(lambda doc: [str(w).lower() for w in doc])

        if self.drop_email:
            if not self.regex_email:
                self.regex_email = r"[\w.-]+@[\w-]+\.[\w.-]+"

            self._log(" --> Removendo e-mails dos documentos.", 2)
            drop_regex(self.regex_email)

        if self.drop_url:
            if not self.regex_url:
                self.regex_url = r"https?://\S+|www\.\S+"

            self._log(" --> Removendo links de URL dos documentos.", 2)
            drop_regex(self.regex_url)

        if self.drop_html:
            if not self.regex_html:
                self.regex_html = r"<.*?>"

            self._log(" --> Removendo tags HTML dos documentos.", 2)
            drop_regex(self.regex_html)

        if self.drop_emoji:
            if not self.regex_emoji:
                self.regex_emoji = r":[a-z_]+:"

            self._log(" --> Removendo emojis dos documentos.", 2)
            drop_regex(self.regex_emoji)

        if self.drop_number:
            if not self.regex_number:
                self.regex_number = r"\b\d+\b"

            self._log(" --> Removendo números dos documentos.", 2)
            drop_regex(self.regex_number)

        if self.drop_punctuation:
            self._log(" --> Removendo pontuação do texto.", 2)
            trans_table = str.maketrans("", "", punctuation)  # Tabela de tradução
            if isinstance(Xt[corpus].iloc[0], str):
                func = lambda doc: doc.translate(trans_table)
            else:
                func = lambda doc: [str(w).translate(trans_table) for w in doc]
            Xt[corpus] = Xt[corpus].apply(func)

        # Remove tokens vazios de cada documento
        if not isinstance(Xt[corpus].iloc[0], str):
            Xt[corpus] = Xt[corpus].apply(lambda doc: [w for w in doc if w])

        return self._convert(Xt)


@beartype
class TextNormalizer(TransformerMixin, OneToOneFeatureMixin):
    """Normaliza o corpus.

    Converte palavras para um padrão mais uniforme. As transformações
    são aplicadas na coluna chamada `corpus`, na mesma ordem em que os
    parâmetros são apresentados. Se não houver uma coluna com esse nome,
    uma exceção é lançada. Se os documentos fornecidos forem strings,
    as palavras são separadas por espaços.

    Esta classe pode ser acessada no experionml através do método [textnormalize]
    [experionmlclassifier-textnormalize]. Leia mais no [guia do usuário]
    [text-normalization].

    Parâmetros
    ----------
    stopwords: bool or str, default=True
        Se deve remover um dicionário predefinido de stopwords.

        - If False: Não remover nenhuma stopword predefinida.
        - If True: Remover stopwords em inglês predefinidas do texto.
        - If str: Idioma de `nltk.corpus.stopwords.words`.

    custom_stopwords: sequence or None, default=None
        Stopwords personalizadas a serem removidas do texto.

    stem: bool or str, default=False
        Se deve aplicar stemming usando [SnowballStemmer][].

        - If False: Não aplicar stemming.
        - If True: Aplicar stemmer baseado no idioma inglês.
        - If str: Idioma de `SnowballStemmer.languages`.

    lemmatize: bool, default=True
        Se deve aplicar lematização usando WordNetLemmatizer.

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não imprimir nada.
        - 1 para imprimir informações básicas.
        - 2 para imprimir informações detalhadas.

    Atributos
    ----------
    feature_names_in_: np.ndarray
        Nomes das features vistos durante o `fit`.

    n_features_in_: int
        Número de features vistos durante o `fit`.

    Veja também
    --------
    experionml.nlp:TextCleaner
    experionml.nlp:Tokenizer
    experionml.nlp:Vectorizer

    Exemplos
    --------
    === "experionml"
        ```pycon
        from experionml import ExperionMLClassifier

        X = [
           ["I àm in ne'w york"],
           ["New york is nice"],
           ["new york"],
           ["hi there this is a test!"],
           ["another line..."],
           ["new york is larger than washington"],
           ["running the test"],
           ["this is a test"],
        ]
        y = [1, 0, 0, 1, 1, 1, 0, 0]

        experionml = ExperionMLClassifier(X, y, test_size=2, random_state=1)
        print(experionml.dataset)

        experionml.textnormalize(stopwords="english", lemmatize=True, verbose=2)

        print(experionml.dataset)
        ```

    === "stand-alone"
        ```pycon
        from experionml.nlp import TextNormalizer

        X = [
           ["I àm in ne'w york"],
           ["New york is nice"],
           ["new york"],
           ["hi there this is a test!"],
           ["another line..."],
           ["new york is larger than washington"],
           ["running the test"],
           ["this is a test"],
        ]

        textnormalizer = TextNormalizer(
            stopwords="english",
            lemmatize=True,
            verbose=2,
        )
        X = textnormalizer.transform(X)

        print(X)
        ```

    """

    def __init__(
        self,
        *,
        stopwords: Bool | str = True,
        custom_stopwords: Sequence[str] | None = None,
        stem: Bool | str = False,
        lemmatize: Bool = True,
        verbose: Verbose = 0,
    ):
        super().__init__(verbose=verbose)
        self.stopwords = stopwords
        self.custom_stopwords = custom_stopwords
        self.stem = stem
        self.lemmatize = lemmatize

    def transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Normaliza o texto.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de features com shape=(n_samples, n_features). Se X não
            for um dataframe, deve ser composto por uma única feature
            contendo os documentos de texto.

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        Retorna
        -------
        dataframe
            Corpus transformado.

        """

        def pos(tag: str) -> wordnet.ADJ | wordnet.ADV | wordnet.VERB | wordnet.NOUN:
            """Obtém a classe gramatical a partir de uma tag.

            Parâmetros
            ----------
            tag: str
                Tag wordnet correspondente a uma palavra.

            Retorna
            -------
            ADJ, ADV, VERB or NOUN
                Classe gramatical da palavra.

            """
            if tag in ("JJ", "JJR", "JJS"):
                return wordnet.ADJ
            elif tag in ("RB", "RBR", "RBS"):
                return wordnet.ADV
            elif tag in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):
                return wordnet.VERB
            else:  # "NN", "NNS", "NNP", "NNPS"
                return wordnet.NOUN

        from nltk import pos_tag
        from nltk.corpus import stopwords, wordnet
        from nltk.stem import SnowballStemmer, WordNetLemmatizer

        Xt = to_df(X, columns=getattr(self, "feature_names_in_", None))
        corpus = get_corpus(Xt)

        self._log("Normalizando o corpus...", 1)

        # Se o corpus não estiver tokenizado, separar por espaço
        if isinstance(Xt[corpus].iloc[0], str):
            Xt[corpus] = Xt[corpus].apply(lambda row: row.split())

        stop_words = set()
        if self.stopwords:
            if isinstance(self.stopwords, bool_t):
                self.stopwords = "english"

            # Obtém stopwords da biblioteca NLTK
            check_nltk_module("corpora/stopwords", quiet=self.verbose < 2)
            stop_words = set(stopwords.words(self.stopwords.lower()))

        # Une stopwords predefinidas com personalizadas
        if self.custom_stopwords is not None:
            stop_words = stop_words | set(self.custom_stopwords)

        if stop_words:
            self._log(" --> Removendo stopwords.", 2)
            f = lambda row: [word for word in row if word not in stop_words]
            Xt[corpus] = Xt[corpus].apply(f)

        if self.stem:
            if isinstance(self.stem, bool_t):
                self.stem = "english"

            self._log(" --> Aplicando stemming.", 2)
            ss = SnowballStemmer(language=self.stem.lower())
            Xt[corpus] = Xt[corpus].apply(lambda row: [ss.stem(word) for word in row])

        if self.lemmatize:
            self._log(" --> Aplicando lematização.", 2)
            check_nltk_module("corpora/wordnet", quiet=self.verbose < 2)
            check_nltk_module("taggers/averaged_perceptron_tagger", quiet=self.verbose < 2)
            check_nltk_module("corpora/omw-1.4", quiet=self.verbose < 2)

            wnl = WordNetLemmatizer()
            f = lambda row: [wnl.lemmatize(w, pos(tag)) for w, tag in pos_tag(row)]
            Xt[corpus] = Xt[corpus].apply(f)

        return self._convert(Xt)


@beartype
class Tokenizer(TransformerMixin, OneToOneFeatureMixin):
    """Tokeniza o corpus.

    Converte documentos em sequências de palavras. Adicionalmente,
    cria n-gramas (representados por palavras unidas com sublinhados,
    por ex., "New_York") com base na sua frequência no corpus. As
    transformações são aplicadas na coluna chamada `corpus`. Se
    não houver uma coluna com esse nome, uma exceção é lançada.

    Esta classe pode ser acessada no experionml através do método [tokenize]
    [experionmlclassifier-tokenize]. Leia mais no [guia do usuário]
    [tokenization].

    Parâmetros
    ----------
    bigram_freq: int, float or None, default=None
        Limiar de frequência para criação de bigramas.

        - If None: Não criar nenhum bigrama.
        - If int: Número mínimo de ocorrências para criar um bigrama.
        - If float: Fração mínima de frequência para criar um bigrama.

    trigram_freq: int, float or None, default=None
        Limiar de frequência para criação de trigramas.

        - If None: Não criar nenhum trigrama.
        - If int: Número mínimo de ocorrências para criar um trigrama.
        - If float: Fração mínima de frequência para criar um trigrama.

    quadgram_freq: int, float or None, default=None
        Limiar de frequência para criação de quadgramas.

        - If None: Não criar nenhum quadgrama.
        - If int: Número mínimo de ocorrências para criar um quadgrama.
        - If float: Fração mínima de frequência para criar um quadgrama.

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não imprimir nada.
        - 1 para imprimir informações básicas.
        - 2 para imprimir informações detalhadas.

    Atributos
    ----------
    bigrams_: pd.DataFrame
        Bigramas criados e suas frequências.

    trigrams_: pd.DataFrame
        Trigramas criados e suas frequências.

    quadgrams_: pd.DataFrame
        Quadgramas criados e suas frequências.

    feature_names_in_: np.ndarray
        Nomes das features vistos durante o `fit`.

    n_features_in_: int
        Número de features vistos durante o `fit`.

    Veja também
    --------
    experionml.nlp:TextCleaner
    experionml.nlp:TextNormalizer
    experionml.nlp:Vectorizer

    Exemplos
    --------
    === "experionml"
        ```pycon
        from experionml import ExperionMLClassifier

        X = [
           ["I àm in ne'w york"],
           ["New york is nice"],
           ["new york"],
           ["hi there this is a test!"],
           ["another line..."],
           ["new york is larger than washington"],
           ["running the test"],
           ["this is a test"],
        ]
        y = [1, 0, 0, 1, 1, 1, 0, 0]

        experionml = ExperionMLClassifier(X, y, test_size=2, random_state=1)
        print(experionml.dataset)

        experionml.tokenize(verbose=2)

        print(experionml.dataset)
        ```

    === "stand-alone"
        ```pycon
        from experionml.nlp import Tokenizer

        X = [
           ["I àm in ne'w york"],
           ["New york is nice"],
           ["new york"],
           ["hi there this is a test!"],
           ["another line..."],
           ["new york is larger than washington"],
           ["running the test"],
           ["this is a test"],
        ]

        tokenizer = Tokenizer(bigram_freq=2, verbose=2)
        X = tokenizer.transform(X)

        print(X)
        ```

    """

    def __init__(
        self,
        bigram_freq: FloatLargerZero | None = None,
        trigram_freq: FloatLargerZero | None = None,
        quadgram_freq: FloatLargerZero | None = None,
        *,
        verbose: Verbose = 0,
    ):
        super().__init__(verbose=verbose)
        self.bigram_freq = bigram_freq
        self.trigram_freq = trigram_freq
        self.quadgram_freq = quadgram_freq

    def transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Tokeniza o texto.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de features com shape=(n_samples, n_features). Se X não
            for um dataframe, deve ser composto por uma única feature
            contendo os documentos de texto.

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        Retorna
        -------
        dataframe
            Corpus transformado.

        """

        def replace_ngrams(row: list[str], ngram: tuple[str]) -> list[str]:
            """Substitui um n-grama por uma palavra unificada com sublinhados.

            Parâmetros
            ----------
            row: list of str
                Um documento no corpus.

            ngram: tuple of str
                Palavras no n-grama.

            Retorna
            -------
            str
               Documento no corpus com n-gramas unificados.

            """
            sep = "<&&>"  # Separador entre palavras em um n-grama.

            row_c = "&>" + sep.join(row) + "<&"  # Indica palavras com separador
            row_c = row_c.replace(  # Substitui o separador do n-grama por sublinhado
                "&>" + sep.join(ngram) + "<&",
                "&>" + "_".join(ngram) + "<&",
            )

            return row_c[2:-2].split(sep)

        import nltk.collocations as collocations
        from nltk import word_tokenize

        Xt = to_df(X, columns=getattr(self, "feature_names_in_", None))
        corpus = get_corpus(Xt)

        self._log("Tokenizando o corpus...", 1)

        if isinstance(Xt[corpus].iloc[0], str):
            check_nltk_module("tokenizers/punkt", quiet=self.verbose < 2)
            Xt[corpus] = Xt[corpus].apply(lambda row: word_tokenize(row))

        ngrams = {
            "bigrams": collocations.BigramCollocationFinder,
            "trigrams": collocations.TrigramCollocationFinder,
            "quadgrams": collocations.QuadgramCollocationFinder,
        }

        for attr, finder in ngrams.items():
            if frequency := getattr(self, f"{attr[:-1]}_freq"):
                # Busca todos os n-gramas no corpus
                ngram_fd = finder.from_documents(Xt[corpus]).ngram_fd

                if frequency < 1:
                    frequency = int(frequency * len(ngram_fd))

                rows = []
                occur, counts = 0, 0
                for ngram, freq in ngram_fd.items():
                    if freq >= frequency:
                        occur += 1
                        counts += freq
                        Xt[corpus] = Xt[corpus].apply(replace_ngrams, args=(ngram,))
                        rows.append({attr[:-1]: "_".join(ngram), "frequency": freq})

                if rows:
                    # Ordena n-gramas por frequência e adiciona o dataframe como atributo
                    df = pd.DataFrame(rows).sort_values("frequency", ascending=False)
                    setattr(self, f"{attr}_", df.reset_index(drop=True))

                    self._log(f" --> Criando {occur} {attr} em {counts} locais.", 2)
                else:
                    self._log(f" --> Nenhum {attr} encontrado no corpus.", 2)

        return self._convert(Xt)


@beartype
class Vectorizer(TransformerMixin):
    """Vetoriza dados de texto.

    Transforma o corpus em vetores significativos de números. A
    transformação é aplicada na coluna chamada `corpus`. Se
    não houver uma coluna com esse nome, uma exceção é lançada.

    Se strategy="bow" ou "tfidf", as colunas transformadas são nomeadas
    de acordo com a palavra que estão incorporando com o prefixo `corpus_`. Se
    strategy="hashing", as colunas são nomeadas hash[N], onde N representa
    a n-ésima coluna com hash.

    Esta classe pode ser acessada no experionml através do método [vectorize]
    [experionmlclassifier-vectorize]. Leia mais no [guia do usuário]
    [vectorization].

    Parâmetros
    ----------
    strategy: str, default="bow"
        Estratégia com a qual vetorizar o texto. Escolha entre:

        - "[bow][]": Bag of Words (Saco de Palavras).
        - "[tfidf][]": Term Frequency - Inverse Document Frequency.
        - "[hashing][]": Vetorizar para uma matriz de ocorrências de tokens.

    return_sparse: bool, default=True
        Se deve retornar a saída da transformação como um dataframe
        de arrays esparsos. Deve ser False quando houver outras colunas
        em X (além de `corpus`) que não sejam esparsas.

    device: str, default="cpu"
        Dispositivo no qual executar os estimadores. Use qualquer string que
        siga o seletor de filtro [SYCL_DEVICE_FILTER][], por ex.
        `#!python device="gpu"` para usar a GPU. Leia mais no
        [guia do usuário][gpu-acceleration].

    engine: str or None, default=None
        Motor de execução a usar para [estimadores][estimator-acceleration].
        Se None, o valor padrão é utilizado. Escolha entre:

        - "sklearn" (padrão)
        - "cuml"

    verbose: int, default=0
        Nível de verbosidade da classe. Escolha entre:

        - 0 para não imprimir nada.
        - 1 para imprimir informações básicas.
        - 2 para imprimir informações detalhadas.

    **kwargs
        Argumentos de palavra-chave adicionais para o estimador `strategy`.

    Atributos
    ----------
    [strategy]_: sklearn transformer
        Instância do estimador (strategy em minúsculo) usado para vetorizar o
        corpus, por ex., `vectorizer.tfidf` para a estratégia tfidf.

    feature_names_in_: np.ndarray
        Nomes das features vistos durante o `fit`.

    n_features_in_: int
        Número de features vistos durante o `fit`.


    Veja também
    --------
    experionml.nlp:TextCleaner
    experionml.nlp:TextNormalizer
    experionml.nlp:Tokenizer

    Exemplos
    --------
    === "experionml"
        ```pycon
        from experionml import ExperionMLClassifier

        X = [
           ["I àm in ne'w york"],
           ["New york is nice"],
           ["new york"],
           ["hi there this is a test!"],
           ["another line..."],
           ["new york is larger than washington"],
           ["running the test"],
           ["this is a test"],
        ]
        y = [1, 0, 0, 1, 1, 1, 0, 0]

        experionml = ExperionMLClassifier(X, y, test_size=2, random_state=1)
        print(experionml.dataset)

        experionml.vectorize(strategy="tfidf", verbose=2)

        print(experionml.dataset)
        ```

    === "stand-alone"
        ```pycon
        from experionml.nlp import Vectorizer

        X = [
           ["I àm in ne'w york"],
           ["New york is nice"],
           ["new york"],
           ["hi there this is a test!"],
           ["another line..."],
           ["new york is larger than washington"],
           ["running the test"],
           ["this is a test"],
        ]

        vectorizer = Vectorizer(strategy="tfidf", verbose=2)
        X = vectorizer.fit_transform(X)

        print(X)
        ```

    """

    def __init__(
        self,
        strategy: VectorizerStarts = "bow",
        *,
        return_sparse: Bool = True,
        device: str = "cpu",
        engine: Engine = None,
        verbose: Verbose = 0,
        **kwargs,
    ):
        super().__init__(device=device, engine=engine, verbose=verbose)
        self.strategy = strategy
        self.return_sparse = return_sparse
        self.kwargs = kwargs

    def _get_corpus_columns(self) -> list[str]:
        """Obtém os nomes das colunas criadas pelo vetorizador.

        Retorna
        -------
        list of str
            Nomes das colunas.

        """
        if hasattr(self._estimator, "get_feature_names_out"):
            return [f"{self._corpus}_{w}" for w in self._estimator.get_feature_names_out()]
        elif hasattr(self._estimator, "get_feature_names"):
            # Estimadores cuML têm um nome de método diferente (retorna um cudf.Series)
            return [f"{self._corpus}_{w}" for w in self._estimator.get_feature_names().to_numpy()]
        else:
            raise ValueError(
                "O método get_feature_names_out não está disponível para strategy='hashing'."
            )

    def fit(self, X: XConstructor, y: YConstructor | None = None) -> Self:
        """Ajusta aos dados.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de features com shape=(n_samples, n_features). Se X não
            for um dataframe, deve ser composto por uma única feature
            contendo os documentos de texto.

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        Retorna
        -------
        Self
            Instância do estimador.

        """
        Xt = to_df(X)
        self._corpus = get_corpus(Xt)

        self._check_feature_names(Xt, reset=True)
        self._check_n_features(Xt, reset=True)

        # Converte uma sequência de tokens para string separada por espaços
        if not isinstance(Xt[self._corpus].iloc[0], str):
            Xt[self._corpus] = Xt[self._corpus].apply(lambda row: " ".join(row))

        strategies = {
            "bow": "CountVectorizer",
            "tfidf": "TfidfVectorizer",
            "hashing": "HashingVectorizer",
        }

        estimator = self._get_est_class(
            name=strategies[self.strategy],
            module="feature_extraction.text",
        )
        self._estimator = estimator(**self.kwargs)

        self._log("Ajustando Vectorizer...", 1)
        self._estimator.fit(Xt[self._corpus])

        # Adiciona o estimador como atributo da instância
        setattr(self, f"{self.strategy}_", self._estimator)

        return self

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """Obtém os nomes das features de saída para a transformação.

        Parâmetros
        ----------
        input_features: sequence or None, default=None
            Usado apenas para validar os nomes das features com os nomes
            vistos no `fit`.

        Retorna
        -------
        np.ndarray
            Nomes das features transformadas.

        """
        check_is_fitted(self, attributes="feature_names_in_")
        _check_feature_names_in(self, input_features)

        og_columns = [c for c in self.feature_names_in_ if c != self._corpus]
        return np.array(og_columns + self._get_corpus_columns())

    def transform(self, X: XConstructor, y: YConstructor | None = None) -> XReturn:
        """Vetoriza o texto.

        Parâmetros
        ----------
        X: dataframe-like
            Conjunto de features com shape=(n_samples, n_features). Se X não
            for um dataframe, deve ser composto por uma única feature
            contendo os documentos de texto.

        y: sequence, dataframe-like or None, default=None
            Não faz nada. Implementado para continuidade da API.

        Retorna
        -------
        dataframe
            Corpus transformado.

        """
        check_is_fitted(self)

        Xt = to_df(X, columns=self.feature_names_in_)

        self._log("Vetorizando o corpus...", 1)

        # Converte uma sequência de tokens para string separada por espaços
        if not isinstance(Xt[self._corpus].iloc[0], str):
            Xt[self._corpus] = Xt[self._corpus].apply(lambda row: " ".join(row))

        matrix = self._estimator.transform(Xt[self._corpus])
        Xt = Xt.drop(columns=self._corpus)  # Remove coluna do corpus original

        if "sklearn" not in self._estimator.__class__.__module__:
            matrix = matrix.get()  # Converte array esparso cupy de volta para scipy

        if not self.return_sparse:
            self._log(" --> Convertendo a saída para um array completo.", 2)
            matrix = matrix.toarray()
        elif not Xt.empty and not is_sparse(Xt):
            # Raise if there are other columns that are non-sparse
            raise ValueError(
                "Valor inválido para o parâmetro return_sparse. O valor deve "
                "ser False quando X contém colunas não esparsas (além de corpus)."
            )

        if self.strategy != "hashing":
            columns = self._get_corpus_columns()
        else:
            # Hashing não tem palavras para usar como nomes de colunas
            columns = [f"hash{i}" for i in range(1, matrix.shape[1] + 1)]

        return self._convert(merge(Xt, to_df(matrix, index=Xt.index, columns=columns)))
