import json

def json_to_latex_table(json_files, output_file):
    latex_table = ""

    for file_name in json_files:
        with open(file_name, 'r') as f:
            data = json.load(f)

        latex_table += "\\begin{tabular}{|" + " ".join(["c |" for _ in data[0]]) + "}\n"
        latex_table += "\\hline\n"

        # Table headers
        headers = " & ".join(data[0].keys()) + "\\\\\n"
        latex_table += headers
        latex_table += "\\hline\n"

        # Table content
        for row in data:
            latex_table += " & ".join(str(val) for val in row.values()) + "\\\\\n"
            latex_table += "\\hline\n"

        latex_table += "\\end{tabular}\n"
        latex_table += "\n"  # Add a space between tables

    with open(output_file, 'w') as f:
        f.write(latex_table)

# List your JSON files here
json_files = ['final_results1.json', 'verification_results.json']
output_file = 'output.tex'
json_to_latex_table(json_files, output_file)