from core.runtime_state import RuntimeStateManager

import json

runtime_state = RuntimeStateManager()

runtime_state.create(
  user_request="Create a reusable runtime_state for agentic systems.",
  objective=None,
  steps=["what is agentic systems.", "evaluate the research report."],
  automated_id=True
)

while not runtime_state.is_complete:

  context = runtime_state.agent_context()
  print(f"\nCURRENT RUNTIME STATE: \n{json.dumps(context, indent=2)}\n")

  if context["current_step"] is None:
    runtime_state.complete()
    print("\nAll steps executed. No pending steps.")
    break

  runtime_state.complete_step({
      "pages_found": 5,
      "pages": [
        {
          "id": 1,
          "page": "....",
          "source": "wikipedia"
        },
        {
          "id": 2,
          "page": "....",
          "source": "youtube"
        },
        {
          "id": 3,
          "page": "....",
          "source": "openai"
        },
        {
          "id": 4,
          "page": "....",
          "source": "anthrop8c"
        },
        {
          "id": 5,
          "page": "....",
          "source": "google"
        }
      ]
  })

  runtime_state.start_next_step()