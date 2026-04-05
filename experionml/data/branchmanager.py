from __future__ import annotations

import shutil
from collections.abc import Iterator
from copy import copy, deepcopy

import pandas as pd
from beartype import beartype
from joblib.memory import Memory
from sklearn.utils.validation import check_memory

from experionml.data.branch import Branch
from experionml.utils.types import Bool, Int
from experionml.utils.utils import ClassMap, DataContainer


@beartype
class BranchManager:
    """Objeto que gerencia branches.

    Mantém referências a uma série de branches e à branch ativa no
    momento. Além disso, sempre armazena uma branch 'original' com
    o dataset original (anterior a qualquer transformação). As branches
    compartilham uma referência ao conjunto de holdout, não à instância em
    si. Quando um objeto de memory é especificado, branches inativas são
    armazenadas em memória (cache).

    Leia mais no [guia do usuário][branches].

    !!! warning
        Esta classe não deve ser chamada diretamente. O BranchManager é
        criado internamente pelas classes [ExperionMLClassifier][], [ExperionMLForecaster][]
        e [ExperionMLRegressor][].

    Parâmetros
    ----------
    memory: str, [Memory][joblibmemory] or None, default=None
        Local para armazenar branches inativas. Se None, todas as branches
        são mantidas em memória. Este objeto de memory é repassado às
        branches para cache do pipeline.

    Atributos
    ----------
    branches: ClassMap
        Coleção de branches.

    og: [Branch][]
        Branch contendo o dataset original. Pode ser qualquer branch
        em `branches` ou uma branch interna chamada `og`.

    current: [Branch][]
        Branch ativa no momento.

    Veja também
    --------
    experionml.data:Branch

    Exemplos
    --------
    ```pycon
    from experionml import ExperionMLClassifier
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)

    # Inicializa o experionml
    experionml = ExperionMLClassifier(X, y, verbose=2)

    # Treina um modelo
    experionml.run("RF")

    # Muda a branch e aplica escalonamento de features
    experionml.branch = "scaled"

    experionml.scale()
    experionml.run("RF_scaled")

    # Compara os modelos
    experionml.plot_roc()
    ```

    """

    def __init__(self, memory: str | Memory | None = None):
        self.memory = check_memory(memory)

        self.branches = ClassMap()
        self.add("main")

        self._og: Branch | None = None

    def __repr__(self) -> str:
        """Exibe as branches contidas."""
        return f"BranchManager([{', '.join(self.branches.keys())}], og={self.og.name})"

    def __len__(self) -> int:
        """Retorna o número de branches no manager."""
        return len(self.branches)

    def __iter__(self) -> Iterator[Branch]:
        """Itera sobre as branches."""
        yield from self.branches

    def __contains__(self, item: str) -> bool:
        """Verifica se item é uma das branches contidas."""
        return item in self.branches

    def __getitem__(self, item: Int | str) -> Branch:
        """Retorna uma branch."""
        try:
            return self.branches[item]
        except KeyError:
            raise IndexError(
                f"This {self.__class__.__name__} instance has no branch {item}."
            ) from None

    @property
    def og(self) -> Branch:
        """Branch contendo o dataset original.

        Esta branch contém os dados anteriores a qualquer transformação.
        Redireciona para a primeira branch com pipeline vazio (se existir),
        ou para uma branch interna chamada `og`.

        """
        return self._og or next(b for b in self.branches if not b.pipeline.steps)

    @property
    def current(self) -> Branch:
        """Branch ativa no momento."""
        return self._current

    @current.setter
    def current(self, branch: str):
        self._current.store()
        self._current: Branch = self.branches[branch]
        self._current.load()

    @staticmethod
    def _copy_from_parent(branch: Branch, parent: Branch):
        """Copia dados e atributos de uma branch pai para uma nova branch.

        Parâmetros
        ----------
        branch: Branch
            Branch que receberá os dados e atributos.

        parent: Branch
            Branch pai da qual as informações serão copiadas.

        """
        if branch.name == "og" and parent._location and branch._location:
            # Cria uma nova cópia dos dados para a branch og
            parent.store(assign=False)
            shutil.copy(
                parent._location.joinpath(f"{parent}.pkl"),
                branch._location.joinpath(f"{branch}.pkl"),
            )
        elif parent._location:
            # Transfere dados da memória para evitar ter
            # os datasets em memória duas vezes ao mesmo tempo
            parent.store()
            branch._container = parent.load(assign=False)
        else:
            # Copia o dataset em memória
            branch._container = deepcopy(parent._container)

        # Deepcopy do pipeline, mas usa os mesmos estimadores
        branch._pipeline = deepcopy(parent._pipeline)
        for i, step in enumerate(parent._pipeline.steps):
            branch.pipeline.steps[i] = step

        # Copia o mapeamento e atribui outras variáveis
        branch._mapping = copy(parent._mapping)
        for attr in vars(parent):
            if not hasattr(branch, attr):  # Se ainda não foi atribuído...
                setattr(branch, attr, getattr(parent, attr))

    def add(self, name: str, parent: Branch | None = None):
        """Adiciona uma nova branch ao manager.

        Se a branch for chamada de `og` (nome reservado para a branch
        original), ela é criada separadamente e armazenada em memória.

        Parâmetros
        ----------
        name: str
            Nome da nova branch.

        parent: Branch or None, default=None
            Branch pai. Dados e atributos da branch pai são
            copiados para a nova branch.

        """
        if name == "og":
            if not self._og:
                self._og = Branch("og", memory=self.memory)
                self._copy_from_parent(self.og, self.current)
        else:
            # Ignora na primeira chamada de __init__
            if self.branches:
                self.current.store()

            self._current = self.branches.append(Branch(name, memory=self.memory))

            if parent:
                self._copy_from_parent(self.current, parent)

    def fill(self, data: DataContainer, holdout: pd.DataFrame | None = None):
        """Preenche a branch atual com dados.

        Esta chamada reinicia o cálculo do holdout em cache.

        Parâmetros
        ----------
        data: DataContainer
            Novos dados para a branch atual.

        holdout: dataframe or None, default=None
            Conjunto de holdout (se existir).

        """
        self.current._container = data
        if holdout is not None:
            self.current._holdout = holdout

        self.current.__dict__.pop("holdout", None)

    def reset(self, *, hard: Bool = False):
        """Reinicia esta instância ao estado inicial.

        O estado inicial do BranchManager contém uma única branch
        chamada `main` sem dados. Não há referência a uma branch
        original (`og`).

        Parâmetros
        ----------
        hard: bool, default=False
            Se True, limpa completamente o cache.

        """
        self.branches = ClassMap()
        self.add("main", parent=self.og)
        self._og = None

        if hard:
            self.memory.clear(warn=False)
