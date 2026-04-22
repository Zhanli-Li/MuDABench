import textwrap
from prompts.prompt_utils import PromptTemplate

PLAN_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
        ## Initial Prompt
        You are a multi-document problem decomposition and query assistant. The user is provided with a metadata description of a complex task and multiple documents. Each document has a unique metadata identifier, and instead of answering the final question directly, you need to design specialized query templates for each document so that you can extract information from it that is relevant to the overall task and organize these subqueries in a reasonable order. Also you need to generate the query templates based on using the same language as the language of user's task. For example, if the user speaks Chinese, you generate the query template in Chinese.

        ## Question template specification
        1. **Single Document Oriented**: Each sub-question template must focus on a single document to ensure that the required information can be located within that document.
        2. **Semantic Mutual Exclusivity**: different templates should be independent of each other in meaning and not duplicated; populated to be able to ask questions naturally and smoothly and logically self-consistent.
        3. **Metadata Placement**: all available metadata fields are placed by {} in at least one template, while all metadata fields must use naming consistent with the metadata field description when placed in templates.
        4. **Metadata constraints (optional)**: for each sub-question template, metadata can be constrained. If you think that answering a multi-document question requires that this sub-question template restricts a certain (some) metadata, please use the name of that metadata as a label, with all possible values listed in list format within the label. For example: `"restriction": {"year": ["2021", "2022"]}`. If there is no restriction, the field can be left out.

        ## User-supplied information
        * Multi-document task description: {task}
        * Available metadata fields: {metadata_description}

        ## Output format
        Please only output the question template and metadata fields in the following json format without any other explanation.
        ```json
        [
            {
                "subtask": "Format template for subquestion",
                "restriction" : "Metadata restriction (optional)"
            },
            {
                "subtask": "Format template for subquestion",
                "restriction" : "Metadata restriction (optional)"
            }
        ]
        ```
        """
    ).strip()
)

JSON_EXTRACTION_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
        ## Initial Prompt
        You are an expert in structured data extraction, extract structured data related to complex tasks directly from multiple sub-question answered conversations and transform it into json format, the final output json data should be easily scalable when transform similar conversations to json. Also you need to generate schema based on using the same language as the users's task language. For example, if the user speaks Chinese, you generate the query template in Chinese.

        ## Input information
        Complex problem description: {task}
        Complete dialog record of sub-question answers: {multi_conversation}

        ## Processing requirements
        1. extract all specific data values (numbers, options, measurements, etc.) related to the complex task and standardize the units of measurement.
        2. identify variable types and add metadata:
           - Classification variables: list the values that actually occur
           - Ordinal variables: Preserve sequential relationships
           - Quantitative variables: specify the units used
        3. For information not provided by the user, keep it as `null`.
        4. Naming of variables should reflect the meaning of the variable and the unit of measure.
        5. ideal json structrue must just have one level, no nested structure and can be easily analysis by python code.
        6. Don't directly include original conversation records in the json.

        ## Output format
        Please output data surrounded by <json>... </json> tags to enclose the data of the json (must be a json of list type (`list[dict]`), where each round of dialog is an element of the list), and with <des>... </des> a brief schema description.
        """
    ).strip()
)

JSON_CONTINUE_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
        ## Initial Prompt
        You are a json continuation helper that converts new conversation records to json data based on the original conversation record converted json data (`list[dict]`) format provided by the user, which is easy to merge with the original data.

        ## Input information
        json data of the original conversation record: {json}
        new conversation record: {new_conversation}

        ## Processing requirements
        1. Convert the new conversation record to json format, making sure it is consistent with the original data structure.
        2. Maintain consistency in variable naming and units of measure.
        3. Make sure the new json data can be seamlessly connected to the original data.

        ## Output Format
        Please enclose your extended section surrounded by <json>...</json> tag, the format of this extension (`list[dict]`) must be able to connect directly to the original json data using `+` in python.
        """
    ).strip()
)

CODE_ACT_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
        ## Initial Prompt
        You are a question answering expert, the user will provide a complex task and multiple copies of related json data and their paths, you need to write code based on this data to get the data needed to answer the question.

        ## Input information
        1. task description: {task}
        2. available json data (few shot): {json_data}
        3. path to the json data: {json_path}
        4. schema of the json data: {json_schema}

        ## Processing requirements
        1. Analyze the task description to identify key issues and data requirements.
        2. Based on the json data provided, write executable python code to read the json data from the user-provided path and extract the required information.
        3. The code output should be readable, ideally the code output should answer the task directly.

        ## Output format
        In your output, you must wrap your code in <execute>... </execute> tags.
        """
    ).strip()
)

ANSWER_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
        ## Initial Prompt
        You are a question answering expert. The user is provided with a complex task, json data summary, analysis code, and code outputs. Generate a final answer in the same language as the user task.

        ## Input information
        1. task description: {task}
        2. json data description: {data}
        3. python code to analyze json data: {code}
        4. analyze result: {code_resp}
        """
    ).strip()
)
