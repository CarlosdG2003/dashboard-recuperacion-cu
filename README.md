# Dashboard Recuperación CU

## Función
Este dashboard permite realizar un análisis de discrepancias entre la recuperación calculada y la recuperación reportada por el sistema SCADA en el proceso de molienda/beneficio de cobre.

### Fórmula utilizada
La recuperación calculada se determina con la siguiente expresión:

Recuperación = [[(COU1CD2001 - COU1RD001) * COU1CF001] / [ (COU1CF001 - COU1RD001) * COU1CD2001]] *100

### Tags requeridos
- `COU1CD2001CU`  
- `COU1RD001CU` (se divide por 10,000)  
- `COU1CF001CU` 
- `COU1-RCT-CU` → Recuperación sistema SCADA  


## Instalación

### Requisitos
- Python 3.13.7
- Docker Desktop

