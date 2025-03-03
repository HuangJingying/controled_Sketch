
import sys, os,json
path = sys.argv[1]
paths =  os.listdir(path)

jsonfile ="metadata.jsonl"

os.chdir(path)
#with open(jsonfile,'w') as json_file:
#    for f in files:
#        file_data = {"file_name": f, "caption": "line art" }
#        json.dump(file_data,json_file )
#        json_file.write("\n")
        
        
        
with open(jsonfile,'w') as json_file:
    for sub in paths:
        #print(sub,os.path.isdir(sub))
        if os.path.isdir(sub):
          fs = os.listdir(os.path.join("",sub))
          for f in fs:
            #print(f)
            filepath = os.path.join(sub,f)
            #print(filepath)
            file_data = {"file_name": filepath, "caption": "abstract sketch of a "+sub }
            json.dump(file_data,json_file )
            json_file.write("\n")