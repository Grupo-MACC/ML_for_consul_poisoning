# 📊 Diccionario de Columnas - Dataset TLS Discovery

## 🔗 Información de Conexión

| Columna | Descripción |
|---------|-------------|
| `ts` | Timestamp Unix de la conexión |
| `id.orig_h` | IP origen (cliente) |
| `id.orig_p` | Puerto origen |
| `id.resp_h` | IP destino (servidor) |
| `id.resp_p` | Puerto destino |
| `proto` | Protocolo (TCP/UDP) |
| `service` | Servicio detectado |
| `conn_state` | Estado de la conexión Zeek |

---

## 📦 Métricas de Bytes

| Columna | Descripción |
|---------|-------------|
| `orig_bytes` | Bytes enviados por origen |
| `resp_bytes` | Bytes enviados por respuesta |
| `bytes_ratio` | Ratio `orig_bytes / resp_bytes` |
| `missed_bytes` | Bytes perdidos en captura |
| `orig_pkts` | Paquetes del origen |
| `orig_ip_bytes` | Bytes IP del origen |
| `resp_pkts` | Paquetes de respuesta |
| `resp_ip_bytes` | Bytes IP de respuesta |

---

## ⏱️ Métricas Temporales

| Columna | Descripción |
|---------|-------------|
| `duration` | Duración de la conexión (segundos) |
| `duration_zscore` | Z-score de la duración (desviación respecto a la media) |
| `conn_interval` | Intervalo desde la conexión anterior |
| `time_since_last_conn` | Tiempo desde última conexión de esta IP |
| `hour_of_day` | Hora del día (0-23) |

---

## 📈 Contadores de Conexiones

| Columna | Descripción |
|---------|-------------|
| `conn_count_10s` | Conexiones en últimos 10 segundos |
| `conn_count_60s` | Conexiones en último minuto |
| `conn_count_300s` | Conexiones en últimos 5 minutos |
| `total_conn_from_ip` | Total conexiones históricas de esta IP |
| `conn_state_encoded` | Estado de conexión codificado numéricamente |

---

## 🔥 Métricas de Comportamiento

| Columna | Descripción |
|---------|-------------|
| `interval_stddev` | Desviación estándar de intervalos |
| `burst_score` | Puntuación de ráfaga (conexiones rápidas) |
| `recon_pattern_score` | Puntuación de patrón de reconocimiento |
| `recent_activity_score` | Puntuación de actividad reciente |

---

## 🔐 Métricas TLS/JA3

| Columna | Descripción |
|---------|-------------|
| `ja3` | Fingerprint JA3 del cliente TLS |
| `ja3s` | Fingerprint JA3S del servidor TLS |
| `ja3_frequency` | Frecuencia de aparición del JA3 |
| `ja3_is_known` | 1 = JA3 conocido/legítimo, 0 = desconocido |
| `ja3_behavior_score` | Puntuación de comportamiento del JA3 |
| `unique_ja3_from_ip` | JA3s únicos usados desde esta IP |

---

## 🖥️ Métricas de IP/Host

| Columna | Descripción |
|---------|-------------|
| `is_known_ip` | 1 = IP conocida/interna, 0 = desconocida |
| `ip_first_seen_hours_ago` | Horas desde primera vez vista esta IP |

---

## 🐳 Métricas Docker

| Columna | Descripción |
|---------|-------------|
| `recent_docker_event` | 1 = evento Docker reciente, 0 = no |
| `time_since_container_start` | Tiempo desde inicio del contenedor (horas) |

---

## 🎯 Etiquetas (Target)

| Columna | Descripción |
|---------|-------------|
| `is_attack` | **0 = Normal, 1 = Ataque** (variable objetivo) |
| `attack_phase` | Fase del ataque: `normal`, `recon`, `exploit`, etc. |

---

## 💡 Notas Rápidas

- **Z-score**: Valores > 2 o < -2 son anómalos
- **burst_score**: Mayor = más conexiones en ráfaga
- **JA3**: Fingerprint único del cliente TLS (útil para detectar herramientas)
- **conn_state_encoded**: Mapeo numérico de estados Zeek (SF=3, S0=1, etc.)
