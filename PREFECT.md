prefect server start          # запустить сервер
prefect config view           # посмотреть текущие настройки

prefect flow-run ls           # список запусков
prefect deployment ls         # список deployment'ов

prefect profile create local
prefect profile use local
prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"

prefect work-pool create local --type process
prefect worker start --pool local

prefect deploy -n dataset-generation --no-prompt
prefect deploy -n lora-adapter_training --no-prompt

prefect deployment inspect "lora-dataset-generation/dataset-generation"