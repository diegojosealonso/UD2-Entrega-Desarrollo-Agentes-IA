import os
import json
import time
import random
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Error crítico: No se encontró la variable GEMINI_API_KEY en el archivo .env")

client = genai.Client(api_key=api_key)

esquema_ticket = {
    "type": "OBJECT",
    "properties": {
        "categoria": {
            "type": "STRING",
            "enum": ["incidencia_envio", "devolucion", "consulta_producto", "otro"],
            "description": "Clasificación temática de la consulta del cliente."
        },
        "prioridad": {
            "type": "STRING",
            "enum": ["alta", "media", "baja"],
            "description": "Nivel de urgencia del ticket."
        },
        "respuesta_sugerida": {
            "type": "STRING",
            "description": "Borrador de respuesta oficial, cordial y empática redactada en español para enviar al cliente."
        }
    },
    "required": ["categoria", "prioridad", "respuesta_sugerida"]
}

config_generacion = types.GenerateContentConfig(
    system_instruction=(
        "Eres un agente experto de atención al cliente de una empresa de comercio internacional. "
        "Analiza el mensaje del cliente (puede venir en español o en inglés), clasifica su categoría "
        "y prioridad, y redacta siempre una respuesta sugerida cordial en español."
    ),
    temperature=0.2,
    response_mime_type="application/json",
    response_schema=esquema_ticket
)

def procesar_ticket_con_reintentos(mensaje_cliente: str, max_retries: int = 5, base_delay: float = 1.0) -> dict:
    """
    Envía la consulta a Gemini aplicando retroceso exponencial con jitter
    ante errores temporales (429, 500, 503).
    """
    codigos_transitorios = {429, 500, 503}

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=f"Analiza y responde a este mensaje de cliente:\n\n{mensaje_cliente}",
                config=config_generacion
            )
            
            resultado_json = json.loads(response.text)
            return resultado_json

        except APIError as e:
            if e.code not in codigos_transitorios or attempt == max_retries - 1:
                print(f"[Error definitivo {e.code}] No se puede completar la petición: {e.message}")
                raise e

            backoff_delay = base_delay * (2 ** attempt)
            jitter = random.uniform(0, backoff_delay)
            sleep_time = backoff_delay + jitter

            print(
                f"[Error temporal {e.code}] Reintento {attempt + 1}/{max_retries}. "
                f"Esperando {sleep_time:.2f}s (Backoff: {backoff_delay:.2f}s + Jitter: {jitter:.2f}s)..."
            )
            time.sleep(sleep_time)

if __name__ == "__main__":
    tickets_prueba = [
        "Hi! My order #88412 was supposed to arrive 5 days ago and the tracking link hasn't updated. I need this urgently for an event!",
        "Hola, compré una chaqueta talla M pero me queda un poco grande. ¿Cómo puedo tramitar el cambio o devolución?"
    ]

    for i, ticket in enumerate(tickets_prueba, start=1):
        print(f"\n--- Procesando Ticket #{i} ---")
        print(f"Entrada: {ticket}")
        try:
            respuesta = procesar_ticket_con_reintentos(ticket)
            print("Salida JSON estructurada:")
            print(json.dumps(respuesta, indent=2, ensure_ascii=False))
        except Exception as err:
            print(f"Fallo al procesar el ticket #{i}: {err}")