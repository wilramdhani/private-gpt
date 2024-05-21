# Private GPT Chatbot - Installation and Usage Guide

Talita Chatbot is an AI-powered conversational agent that can answer questions based on text data extracted from PDF files. Follow the instructions below to install and run the Python script.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/wilramdhani/private-gpt.git
   ```

2. Navigate to the project directory:

   ```bash
   cd private-gpt
   ```

3. Create and activate a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # For Linux/Mac
   # Or
   venv\Scripts\activate  # For Windows
   ```

4. Install the required dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the project directory and add the following content:

   ```plaintext
   LLM_URL=xxxxxxx
   ```

   Replace the URL with the actual URL of your Language Model.

## Usage

1. Ensure that you have placed the PDF files you want to process in a directory.
2. Open a terminal or command prompt.

3. Navigate to the project directory:

   ```bash
   cd path/to/private-gpt
   ```

4. Run the Python script:

   ```bash
   python private-gpt.py
   ```

5. Enter your questions when prompted. The script will provide responses based on the content of the PDF files.

6. Type `exit` or `quit` to stop the script.

## Additional Notes

- You can customize the behavior of the chatbot by modifying the Python script (`private-gpt.py`) according to your requirements.
- Remember to update the LLM URL in the `.env` file if necessary.
- If you encounter any issues or have questions, feel free to open an issue on the GitHub repository.
