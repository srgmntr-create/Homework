# Homework sobre SVMs - Inteligência Computacional

Este repositório contém a resolução do Homework sobre Support Vector Machines (SVMs) para a disciplina de Inteligência Computacional.

**Aluno:** Sérgio S. Monteiro
**Professor:** Aldebaro Klautau
**Disciplina:** Inteligência Computacional
**Data de Entrega:** 23/05/2025

## Descrição do Projeto

Este trabalho aborda diversos aspectos teóricos e práticos relacionados aos classificadores SVM, incluindo:
* Análise visual de regiões de decisão de diferentes kernels de SVM.
* O papel da regularização (parâmetro C) e seu impacto no número de vetores de suporte.
* Estimativa da redução de custo computacional ao converter uma SVM linear para um perceptron.
* Interpretação detalhada dos parâmetros de modelos SVM lineares e não-lineares treinados (vetores de suporte, coeficientes duais, bias).
* Implementação prática do treinamento de SVMs lineares e com kernel RBF usando a biblioteca scikit-learn em Python.
* Normalização de dados e sua importância no treinamento de SVMs.
* Otimização de hiperparâmetros (C e gamma) para SVMs com kernel RBF utilizando um conjunto de validação.
* Comparação entre a função de decisão calculada manualmente e a fornecida pelo scikit-learn.

## Conteúdo do Repositório

* **`Homework_SVMs_Sergio_Monteiro.ipynb`**: O Jupyter Notebook contendo todas as resoluções das questões, incluindo as explicações em Markdown e os blocos de código Python executáveis.
* **`dataset_train.txt`**: Conjunto de dados utilizado para o treinamento dos modelos.
* **`dataset_test.txt`**: Conjunto de dados utilizado para o teste dos modelos.
* **`dataset_validation.txt`**: Conjunto de dados utilizado para a validação e otimização de hiperparâmetros da SVM com kernel RBF (Questão 7).
* **`requirements.txt`**: Arquivo listando as dependências Python necessárias para executar o notebook.
* **`README.md`**: Este arquivo.

## Instruções para Execução

Para visualizar e executar o notebook:

1.  **Clone o Repositório (Opcional):**
    ```bash
    git clone [https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content)
    cd [Nome da pasta do repositório]
    ```
    Alternativamente, baixe o ZIP do projeto e extraia os arquivos.

2.  **Ambiente Virtual (Recomendado):**
    É uma boa prática criar um ambiente virtual para isolar as dependências do projeto.
    ```bash
    python -m venv svm_env
    # Para Windows:
    # svm_env\Scripts\activate
    # Para macOS/Linux:
    # source svm_env/bin/activate
    ```

3.  **Instale as Dependências:**
    Navegue até a pasta do projeto (onde o arquivo `requirements.txt` está localizado) e execute:
    ```bash
    pip install -r requirements.txt
    ```
    As principais bibliotecas utilizadas são `numpy` e `scikit-learn`.

4.  **Inicie o Jupyter Notebook ou Jupyter Lab:**
    No terminal, dentro da pasta do projeto, execute:
    ```bash
    jupyter notebook
    ```
    ou
    ```bash
    jupyter lab
    ```

5.  **Abra o Notebook:**
    No navegador, abra o arquivo `Homework_SVMs_Sergio_Monteiro.ipynb`.

6.  **Execute as Células:**
    * Você pode executar cada célula individualmente (usando Shift + Enter).
    * Para executar todo o notebook de uma vez e reproduzir todos os resultados, vá ao menu "Kernel" e selecione "Restart & Run All".

**Observações:**
* Os arquivos de dataset (`dataset_train.txt`, `dataset_test.txt`, `dataset_validation.txt`) devem estar no mesmo diretório que o arquivo `.ipynb` para que o código de carregamento funcione como está.
* O notebook foi salvo com as saídas das células já visíveis para facilitar a correção.

## Estrutura do Notebook

O notebook está organizado da seguinte forma:
* **Questões 1 a 5:** Resoluções teóricas e interpretações, apresentadas em células Markdown.
* **Questão 6:** Implementação de uma SVM linear, incluindo carregamento de dados, normalização, treinamento, avaliação e análise de custo computacional. Apresenta células de Markdown para explicação e células de código Python para a implementação.
* **Questão 7:** Projeto de uma SVM com kernel RBF, incluindo otimização de hiperparâmetros, descrição do modelo escolhido e verificação da função de decisão. Apresenta células de Markdown para explicação e células de código Python para a implementação.

Quaisquer dúvidas ou problemas, favor entrar em contato.
