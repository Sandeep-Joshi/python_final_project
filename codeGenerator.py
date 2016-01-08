import os
# -*- coding: utf-8 -*-
# -*- coding: iso-8859-15 -*-

code = open("finalCode.py", 'wb')

# Include code from template
read = open("template.py", 'rb')
for line in read:
    code.write(line)
code.write('\n\n')
code.write("##<<--.-->>##\n") # marks end of code

# Include all dump files
for item in os.listdir("./"):
    if os.path.isfile(os.path.join("./", item)):
        # for root, dirs, item in os.walk("./"):
        i = 0
        # if dirs != os.getcwd():
        #     print dirs, os.getcwd()
        #     continue
        # print "In dir - " + str(dirs)
        # for name in files:
        if item.endswith((".pkl", ".npy")):
            print "Adding file: " + item
            i += 1
            read = open(item, 'rb')
            #Write the name of the file to prog
            code.write('namedump{0} = "{1}"\n'.format(str(i), item))
            code.write('valuedump{0} = """\n'.format(str(i)))
            for line in read:
                code.write(line)
            code.write('"""\n')

read.close()
code.close()
