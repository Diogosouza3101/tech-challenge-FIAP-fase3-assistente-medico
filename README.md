# Tech Challenge - Fase 3 - Assistente Clínico com LLM

## 1. Descrição do Projeto

Projeto acadêmico da pós-graduação em IA para DEVs que implementa um assistente clínico de apoio à decisão com uma LLM customizada. A solução combina o modelo **Qwen2.5-3B-Instruct**, um adapter treinado com QLoRA, consulta a pacientes sintéticos em SQLite e recuperação de protocolos fictícios por RAG.

O desenvolvimento e os resultados experimentais estão registrados nos notebooks `00` a `10`. A aplicação modular em `src/` reutiliza a lógica dos notebooks `07`, `08`, `09` e `10` e oferece uma interface Streamlit.

## 2. Problema

O cenário simula um hospital no qual o profissional precisa reunir dados do paciente e consultar protocolos internos antes de analisar um caso. As informações incluem histórico familiar, resultado de exame já realizado, exame pendente e status de acompanhamento.

Uma LLM de uso geral pode acrescentar informações ausentes, confundir exames pendentes com resultados disponíveis ou produzir orientações fora do escopo pretendido. O projeto explora como contextualização, ajuste supervisionado e verificação determinística podem apoiar uma consulta mais controlada e rastreável.

## 3. Objetivo

Construir uma demonstração que permita consultar um paciente por ID, recuperar protocolos pertinentes e gerar uma resposta contextualizada, indicando as fontes utilizadas e submetendo o texto a uma verificação de possível prescrição ou dosagem. A decisão permanece com o profissional médico responsável.

## 4. Arquitetura da Solução

```text
Usuário → Streamlit → LangGraph → SQLite → RAG/FAISS
        → Qwen Fine-Tuned → Guardrail → Logging → Resposta
```

O LangGraph coordena as etapas. O SQLite fornece somente o registro solicitado; o RAG combina a pergunta e o contexto desse paciente para recuperar protocolos. O Qwen base quantizado recebe o adapter LoRA local e gera a resposta. O guardrail aprova o texto ou o substitui por uma mensagem de bloqueio. Após o registro em log, o Streamlit apresenta o resultado.

Modelo e índice vetorial são carregados sob demanda e reutilizados em memória. O grafo não utiliza persistência de conversas entre pacientes. Na aplicação atual, a geração é determinística (`do_sample=False`), com limite de `max_new_tokens=400`. A interface renderiza Markdown e remove apenas a seção final vazia “Pontos de atenção”, inclusive quando seu Markdown está incompleto.

## 5. Tecnologias Utilizadas

| Tecnologia | Uso no projeto |
| --- | --- |
| Python | Notebooks e aplicação modular; notebooks registrados com Python 3.11 |
| PyTorch | Execução do modelo e operações em GPU |
| Hugging Face Transformers | Tokenizer, chat template, carregamento e geração |
| Qwen/Qwen2.5-3B-Instruct | Modelo base da solução |
| QLoRA/LoRA e PEFT | Ajuste de adapters e carregamento do adapter treinado |
| bitsandbytes e Accelerate | Quantização 4-bit e suporte ao carregamento com `device_map="auto"` |
| LangChain | Documents, integração SQL nos notebooks, embeddings e vector store |
| LangGraph | Estado, nós e rotas condicionais do assistente |
| FAISS | Busca vetorial dos protocolos |
| sentence-transformers | Modelo de embeddings multilíngue |
| SQLite e SQLAlchemy | Banco local; SQLAlchemy é utilizado pela integração SQLDatabase nos notebooks |
| Streamlit | Interface de consulta e apresentação dos resultados |
| pandas e Matplotlib | Preparação/tabulação dos dados e avaliação com gráfico nos notebooks |
| Hugging Face Datasets e TRL | Dataset e SFTTrainer no notebook de treinamento |

`requirements.txt` reúne as bibliotecas usadas pela aplicação e pelos notebooks. As integrações LangChain estão listadas nos pacotes efetivamente importados: `langchain-core`, `langchain-community` e `langchain-huggingface`. Módulos da biblioteca padrão, como `sqlite3`, `pathlib`, `json`, `re` e `logging`, não exigem instalação separada.

## 6. Dataset

Foram utilizados **dados sintéticos e protocolos institucionais fictícios**, sem dados reais de pacientes.

- `data/raw/pacientes.csv`: quatro pacientes sintéticos, identificados de `PAC001` a `PAC004`.
- `data/raw/protocolos_medicos.json`: cinco protocolos, de `PROTO-001` a `PROTO-005`.
- `data/database/hospital.db`: banco SQLite com a tabela `pacientes`.
- `data/processed/fine_tuning_medico.jsonl`: 60 exemplos curados, em formato conversacional, distribuídos em dez categorias.
- `data/processed/train.jsonl` e `validation.jsonl`: 40 exemplos de treino e 20 de validação.

O notebook `02` combina três perguntas e duas respostas por categoria. A curadoria está expressa nas perguntas, respostas e regras redigidas no próprio notebook; não representa validação clínica externa. O notebook `03` separa as perguntas por categoria e registra **zero perguntas idênticas compartilhadas entre treino e validação**. As categorias e os modelos de resposta continuam próximos entre as partições.

## 7. Fine-Tuning

O [notebook 05](05_finetuning_qlora.ipynb) registra o treinamento local em uma **GPU NVIDIA GeForce RTX 3060**. O [notebook 00](00_teste_ambiente.ipynb) registra 12 GB de VRAM.

| Configuração | Valor registrado |
| --- | --- |
| Modelo base | `Qwen/Qwen2.5-3B-Instruct` |
| Quantização | 4-bit NF4 com double quantization |
| Tipo de cálculo | `torch.bfloat16` |
| Técnica | QLoRA, com adapter LoRA via PEFT |
| Épocas | 3 |
| Treino / validação | 40 / 20 exemplos |
| Rank LoRA / alpha / dropout | 16 / 32 / 0,05 |
| Módulos alvo no treinamento | `all-linear` |
| Batch por dispositivo / acumulação | 1 / 4 passos |
| Learning rate | `1e-4` |
| Comprimento máximo no treinamento | 512 tokens |
| Parâmetros treináveis | 29.933.568, equivalentes a 0,9607% dos parâmetros reportados |

O notebook utiliza gradient checkpointing e avaliação ao final de cada época. O adapter e o tokenizer foram salvos em `models/qwen2.5-3b-medical-lora/`.

A aplicação apenas carrega o modelo base quantizado, aplica o adapter existente e executa `model.eval()`. **Abrir o Streamlit não realiza novo fine-tuning.**

## 8. Avaliação

Os [notebooks 04](04_modelo_base_qwen.ipynb) e [06](06_avaliacao_finetuned.ipynb) registram a comparação entre modelo base e fine-tuned. A comparação utiliza dez perguntas nas categorias alteração suspeita, histórico familiar, exames pendentes, prescrição, dosagem, diagnóstico definitivo, informação insuficiente, rastreabilidade, acompanhamento e encaminhamento.

O notebook `06` aplica uma avaliação manual de quatro critérios, cada um com nota de 0 a 2: segurança, validação médica, aderência ao protocolo e rastreabilidade. O máximo é de oito pontos por caso, totalizando 80 pontos.

| Modelo | Pontuação manual | Percentual da pontuação máxima |
| --- | --- | --- |
| Base | 48/80 | 60,00% |
| Fine-Tuned | 53/80 | 66,25% |

A diferença registrada é de **6,25 pontos percentuais**. Esses valores são notas da rubrica manual dos dez casos, **não acurácia diagnóstica nem comprovação de segurança clínica**.

No teste avulso que solicita um medicamento e sua dose, a saída salva no notebook `04` começa com uma recusa, mas depois apresenta nomes de medicamentos e doses. Para a mesma solicitação, o notebook `06` registra a recusa do fine-tuned em prescrever ou indicar doses, mantendo a decisão com o médico. Esse teste é uma evidência pontual; não demonstra que o modelo recusará todas as solicitações inadequadas.

As evidências tabulares estão em:

- [resultados_modelo_base.csv](data/processed/resultados_modelo_base.csv): respostas do modelo base.
- [comparacao_base_vs_finetuned.csv](data/processed/comparacao_base_vs_finetuned.csv): respostas lado a lado.
- [avaliacao_base_vs_finetuned.csv](data/processed/avaliacao_base_vs_finetuned.csv): notas e totais da avaliação manual.

Essa avaliação antecede a integração final com RAG e guardrail. Não é uma medição de desempenho ponta a ponta do aplicativo Streamlit.

## 9. RAG e Protocolos

O arquivo JSON é convertido em `Document` com ID, título, conteúdo e metadados de fonte. Os embeddings são gerados com **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**, via `HuggingFaceEmbeddings`, e indexados em FAISS em memória.

A busca semântica utiliza a pergunta junto aos dados do paciente e recupera inicialmente três documentos (`k=3`). Depois, aplica as inclusões obrigatórias já definidas no notebook `10`.

| Protocolo | Tema e papel no contexto |
| --- | --- |
| PROTO-001 | Avaliação inicial de alteração mamária suspeita; revisão dos achados e avaliação especializada conforme o protocolo fictício |
| PROTO-002 | Histórico familiar e fatores de risco; contextualização sem conclusão diagnóstica isolada |
| PROTO-003 | Exames pendentes; incluído obrigatoriamente quando há pendência no contexto |
| PROTO-004 | Segurança na recomendação clínica; sempre incluído |
| PROTO-005 | Rastreabilidade da resposta; sempre incluído |

As inclusões podem elevar o total acima de três protocolos. O retorno preserva IDs, títulos, conteúdo e o caminho da fonte para exibição.

## 10. LangChain e Banco de Dados

Os notebooks `07`, `08` e `10` usam `SQLDatabase`, da integração LangChain Community, com SQLite/SQLAlchemy. O notebook `07` demonstra a criação da tabela a partir do CSV sintético e sua consulta.

Na aplicação modular, `src/database/database.py` utiliza `sqlite3` com **consulta SQL fixa e parametrizada**, banco aberto somente para leitura e filtro pelo ID solicitado. A LLM não gera nem executa SQL livremente.

O ID é normalizado com remoção de espaços externos e conversão para maiúsculas, sendo validado no formato `PAC` seguido de três dígitos. Os campos retornados são ID, idade, sexo, histórico familiar, resultado do exame, exames pendentes e status. O contexto distingue explicitamente **resultado do exame já realizado** de **exame pendente**.

## 11. LangGraph

```text
START → validar entrada → buscar paciente → recuperar protocolos
      → gerar resposta → validar segurança → registrar log → END
```

O estado transporta a pergunta, o ID, os dados consultados, os protocolos, a resposta e o status. ID inválido, pergunta vazia e paciente inexistente desviam diretamente para o registro de log, sem geração. Uma resposta bloqueada é substituída pela mensagem do módulo de segurança, e o texto bloqueado é removido do estado retornado à interface.

Os resultados previstos são `aprovado`, `bloqueado` ou `erro`. Falhas operacionais, como indisponibilidade do modelo ou do banco, são capturadas pela interface e exibidas como erro de execução; não equivalem a uma aprovação de segurança.

## 12. Segurança

O projeto utiliza duas camadas complementares:

1. **Comportamento aprendido no fine-tuning:** exemplos de recusa a prescrição, dosagem e diagnóstico definitivo, além de instruções sobre dados ausentes e validação médica.
2. **Guardrail determinístico:** verifica números associados a unidades como `mg`, `mcg`, `g`, `ml`, `mg/m²` e `mg/m2`, além das frases “prescrevo”, “recomendo tomar”, “deve tomar” e “a dose recomendada é”, sem distinguir maiúsculas de minúsculas.

Quando encontra um desses padrões, o guardrail retorna `bloqueado` e o motivo “Possível prescrição ou dosagem detectada.” A interface exibe uma mensagem de bloqueio, em vez da resposta original.

O status `aprovado` significa apenas que esses padrões não foram detectados. A ferramenta **não substitui avaliação médica**.

## 13. Explainability e Rastreabilidade

A interface apresenta os dados do paciente efetivamente enviados ao modelo, os protocolos recuperados, suas fontes, a resposta final e o status de segurança. O prompt também instrui o modelo a citar os protocolos utilizados.

O log em `logs/assistant.log` usa **UTF-8**, timestamp, ID do paciente, pergunta, IDs dos protocolos, status e eventual motivo de bloqueio. As rotas de erro de entrada também passam pelo logging.

Essa apresentação permite inspecionar o contexto fornecido à geração, mas não explica os mecanismos internos da rede neural nem garante que cada afirmação esteja sustentada pela fonte. O log não armazena o texto completo da resposta nem constitui, sozinho, uma auditoria clínica completa.

## 14. Estrutura do Projeto

Árvore da estrutura atual, com caches omitidos e o conteúdo interno dos checkpoints resumido:

```text
.
├── .gitignore
├── README.md
├── requirements.txt
├── 00_teste_ambiente.ipynb
├── 01_preparacao_dataset.ipynb
├── 02_geracao_dataset_finetuning.ipynb
├── 03_divisao_dataset.ipynb
├── 04_modelo_base_qwen.ipynb
├── 05_finetuning_qlora.ipynb
├── 06_avaliacao_finetuned.ipynb
├── 07_langchain_banco_pacientes.ipynb
├── 08_assistente_contextualizado.ipynb
├── 09_rag_protocolos.ipynb
├── 10_assistente_rag.ipynb
├── app/
│   └── app.py
├── src/
│   ├── __init__.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── model.py
│   ├── database/
│   │   ├── __init__.py
│   │   └── database.py
│   ├── chains/
│   │   ├── __init__.py
│   │   └── medical_rag.py
│   ├── graph/
│   │   ├── __init__.py
│   │   └── medical_graph.py
│   ├── safety/
│   │   ├── __init__.py
│   │   └── guardrails.py
│   └── utils/
│       ├── __init__.py
│       └── logger.py
├── data/
│   ├── raw/
│   │   ├── pacientes.csv
│   │   └── protocolos_medicos.json
│   ├── processed/
│   │   ├── fine_tuning_medico.jsonl
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   ├── resultados_modelo_base.csv
│   │   ├── comparacao_base_vs_finetuned.csv
│   │   └── avaliacao_base_vs_finetuned.csv
│   └── database/
│       └── hospital.db
├── models/
│   └── qwen2.5-3b-medical-lora/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors     # local, ignorado pelo Git
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       ├── chat_template.jinja
│       ├── README.md
│       ├── checkpoint-20/               # local, ignorado pelo Git
│       └── checkpoint-30/               # local, ignorado pelo Git
├── logs/
│   └── assistant.log                    # local, ignorado pelo Git
├── images/                             # atualmente vazia
└── notebooks/                          # atualmente vazia
```

Os notebooks existentes estão na **raiz**, e não dentro de `notebooks/`. Diretórios vazios não são preservados pelo Git. Nenhum notebook precisa ser movido para executar o aplicativo.

O `.gitignore` mantém datasets sintéticos, banco, notebooks e resultados CSV disponíveis para versionamento. Em `models/`, permite somente os metadados, tokenizer e chat template do adapter conhecido; os pesos e checkpoints permanecem locais. O arquivo local `adapter_model.safetensors` tem aproximadamente 60 MB, e cada `optimizer.pt` dos checkpoints tem aproximadamente 120 MB.

## 15. Como Executar

### Pré-requisitos

- Ambiente Conda `fiap_fase3` previamente criado, com Python 3.11, conforme os notebooks.
- GPU NVIDIA e instalação PyTorch/CUDA compatível com o ambiente de execução. O caminho validado usa quantização 4-bit e BF16; execução sem GPU não foi validada neste projeto.
- Banco `data/database/hospital.db` e protocolos `data/raw/protocolos_medicos.json` disponíveis.
- Adapter treinado e tokenizer disponíveis em `models/qwen2.5-3b-medical-lora/`.
- Acesso aos modelos base e de embeddings no primeiro carregamento, ou arquivos já disponíveis no cache local do Hugging Face.

### Iniciar a aplicação

Na raiz do projeto:

```bash
conda activate fiap_fase3
pip install -r requirements.txt
streamlit run app/app.py
```

As versões em `requirements.txt` foram fixadas a partir dos pacotes realmente instalados no ambiente `fiap_fase3`, com Python 3.11.16. O arquivo não é um lockfile completo das dependências transitivas. As dependências `datasets` e `trl` atendem ao notebook de fine-tuning; instalá-las não inicia treinamento. `jupyterlab==4.6.3` e `ipykernel==7.3.0` estão incluídos para abrir e executar os notebooks interativamente com o kernel desse ambiente.

**Instalação PyTorch/CUDA validada:** o ambiente consultado contém `torch==2.14.0+cu130`, reporta `torch.version.cuda == "13.0"` e `torch.cuda.is_available() == True`. Essa versão exata foi preservada no arquivo de dependências. O índice ou a origem de instalação desse build não foi identificado nesta revisão, portanto nenhum índice de download foi presumido. Em um ambiente novo, esse build precisa estar disponível por sua origem de instalação; o comando `pip install -r requirements.txt` não garante que ele exista no índice padrão. Preserve a instalação PyTorch/CUDA validada. Nenhum pacote foi instalado, atualizado ou rebaixado nesta revisão.

Na interface, informe, por exemplo, `PAC001` e a pergunta: “Analise a situação atual deste paciente e informe quais pontos precisam de atenção.” Clique em **Analisar** e consulte os dados, protocolos, resposta e status apresentados. A primeira consulta válida pode demorar mais por carregar os modelos e criar o índice FAISS.

### Download do adapter Fine-Tuned

**Link para download:** [Adapter LoRA no Hugging Face](https://huggingface.co/Diogossouza/qwen2.5-3b-medical-fiap-lora)

O adapter LoRA treinado está publicado no Hugging Face e pode ser obtido no repositório indicado acima. Para executar a aplicação após clonar o repositório, baixe o **adapter já treinado** e coloque `adapter_model.safetensors` no diretório configurado, junto de `adapter_config.json`, `tokenizer.json`, `tokenizer_config.json` e `chat_template.jinja`.

Clonar apenas os arquivos versionados não recupera os pesos ignorados. O modelo base é carregado separadamente pelo identificador `Qwen/Qwen2.5-3B-Instruct`; o adapter não o substitui. Os checkpoints de treinamento e os estados do otimizador não são necessários para inferência.

Uma futura distribuição dos pesos por armazenamento de artefatos ou Git LFS deve ser configurada explicitamente; nenhum desses mecanismos está configurado por esta documentação. O `.gitignore` não remove arquivos que já tenham sido adicionados ao histórico Git.

Os caminhos da aplicação são resolvidos com `pathlib` a partir da raiz do projeto. Não é necessário executar novamente os notebooks de preparação ou treinamento para usar o banco e o adapter existentes. Os notebooks registram etapas que escrevem datasets, recriam tabelas, treinam ou manipulam logs; sua execução integral não faz parte da inicialização do aplicativo.

## 16. Demonstração

**Vídeo no YouTube:** link a adicionar após a gravação.

A demonstração poderá apresentar uma consulta válida, a exibição de dados e fontes, os tratamentos de entrada inválida/paciente inexistente e o teste determinístico de bloqueio já registrado no notebook `10`.

## 17. Limitações

- Dataset sintético pequeno: quatro pacientes, cinco protocolos fictícios e 60 exemplos de ajuste.
- Cenários educacionais restritos, sem validação em população clínica real.
- Avaliação manual de dez casos, sem evidência estatística de generalização ou comparação controlada de todos os componentes.
- Possibilidade de respostas inadequadas, informações inventadas e truncamento mesmo com limite de 400 novos tokens.
- Guardrail baseado em padrões: pode bloquear menções legítimas a unidades e deixar passar formulações não previstas; não verifica integralmente diagnóstico, correção clínica ou ataques ao prompt.
- Citação e recuperação de protocolos não garantem fidelidade de todas as afirmações às fontes.
- Dependência de recursos locais, GPU, pesos externos e compatibilidade das bibliotecas; não há fallback de inferência validado para CPU.
- Interface sem autenticação ou autorização por usuário, inadequada para exposição pública com dados reais.
- Necessidade de revisão e validação humana de qualquer saída.

## 18. Aviso

**Este sistema é acadêmico e destinado exclusivamente à demonstração educacional. Não deve ser utilizado para diagnóstico, prescrição, definição de doses ou tomada de decisão médica real. Os pacientes e protocolos são fictícios. A ferramenta não substitui a avaliação de um profissional médico.**
