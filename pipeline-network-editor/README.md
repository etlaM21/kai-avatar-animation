## API Request Format

To trigger a generation, send an HTTP POST request to the /generate endpoint.

- Endpoint: http://127.0.0.1:42069/generate

- Headers: Content-Type: application/json

JSON Payload Body Structure
```JSON
{
  "prompt": "A person performing a high-impact roundhouse kick",
  "filename_prefix": "roundhouse_kick"
}
```