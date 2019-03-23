# This piece of code aim to transfer content in sample.csv
# to markdown format so that it is more reader friendly

import csv;

if __name__ == '__main__':
    with open('sample.md', 'w') as mdfile:
        mdfile.write('# Samples\n\n')

        with open('sample.csv') as csvfile:
            samples = csv.reader(csvfile)

            sample_num = 1
            for sample_row in samples:
                mdfile.write('## ' + str(sample_num) + '. Sample ' + sample_row[0] + '\n\n')

                mdfile.write('File Path: ' + sample_row[1] + '\n\n')
                mdfile.write('Comment Line Number: ' + sample_row[3] + '\n\n')
                mdfile.write('Label: ' + sample_row[4] + '\n\n')

                mdfile.write('```c++\n')
                comment = sample_row[2].replace('[comma]', ',').replace('[enter]', '\n')
                mdfile.write(comment)
                mdfile.write('\n```\n\n')

                mdfile.write('```c++\n')
                code = sample_row[5].replace('[comma]', ',').replace('[enter]', '\n')]
                mdfile.write(code)
                mdfile.write('\n```\n\n')

                sample_num += 1

