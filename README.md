# Backend_Profagro

## Описание

Backend реализует промышленный пайплайн для автоматизации калибровки центробежных разбрасывателей удобрений на базе Retrieval-Augmented Generation. Сервисы обеспечивают индексацию, гибридный поиск, генерацию ответов и интеграцию с LLM, а также оркестрацию всех этапов обработки данных.

## Архитектура

- **indexer/** — сервис для массовой загрузки и индексации PDF-руководств и транскрипций видео в Milvus (семантический поиск, embedder: bge-m3) и OpenSearch (лексический поиск).
- **search-api/** — FastAPI-сервис, реализующий гибридный ретривер (OpenSearch + Milvus), Cross-Encoder-реранкер (bge-reranker-v2-m3), обращение к LLM (GPT-4o, GigaChat-MAX-2), формирование финального ответа с точными ссылками на источники.
- **Оркестрация** — Prefect, пайплайны полностью воспроизводимы и масштабируемы.
- **Хранение данных** — Cloud.ru S3 (объектное хранилище для корпуса знаний и эмбеддингов).
- **Инференс embedder/реранкер** — vLLM.

## Технологический стек

- Python 3.10+
- FastAPI, Prefect
- OpenSearch, Milvus
- boto3, PyYAML, pydantic
- Docker, docker-compose
- vLLM

## Запуск

### Через Docker Compose (рекомендуется)
1. Перейдите в корень репозитория
2. Запустите:
   ```bash
   docker-compose up --build
   ```

### Локально (пример для indexer)
1. Установите зависимости:
   ```bash
   cd backend/indexer
   pip install -r requirements.txt
   ```
2. Запустите Prefect flow или скрипт индексации

### Локально (пример для search-api)
1. Установите зависимости:
   ```bash
   cd backend/search-api
   pip install -r requirements.txt
   ```
2. Запустите API:
   ```bash
   uvicorn project.app:app --reload
   ```

## Практическая значимость
- Система внедрена в ООО «ПрофАгро», доказала снижение ошибок дозирования и ускорение настройки техники.
- Пайплайн готов к горизонтальному и вертикальному масштабированию, расширению на новые типы техники и языки.

## Документация
- [indexer/README.md](indexer/README.md)
- [search-api/README.md](search-api/README.md)