import os, sys, argparse,json, time, subprocess
from src.etl import ETL
import xlrd
#xl_workbook = xlrd.open_workbook("repos_properties_sorted.xlsx")
#sheet_names = xl_workbook.sheet_names()


#xl_sheet = xl_workbook.sheet_by_name(sheet_names[0]) # Full list 

etl = ETL(itype='git',iloc="https://github.com/acmeair/acmeair", token=None)
#etl.clean_db() 
#etl.run()
#delete_filesystem = "MATCH p=()-[r:FILESYSTEM]->() DELETE p"
#etl.graph_query(delete_filesystem)


"""
 # Named Edge List  Don't forget : must be re-embedded
statement = 'MATCH (n)RETURN n' # For now, we only uses edges
nodes = etl.graph_query(statement)['nodes']

name_id = {}
for node in nodes:
    name_id[node['id']] = node['properties']['name']

statement = 'MATCH (n)-[r]->(m) RETURN r' # For now, we only uses edges
edges = etl.graph_query(statement)['edges']

for edge in edges:
    print(    str( name_id[edge['start_node']['id']])  + " "  +   str( name_id[edge['end_node']['id']]) + " " +  str(edge['properties']['degree'])    )

#    if edge['type'] != 'FILESYSTEM':
#        f.write(str(edge['start_node']['id']) + " "+  str(edge['end_node']['id']) +" "+ str(edge['properties']['degree']) + "\n" ) 

"""

"""
# get keys processed in some format
louvain = etl.analytics("louvain")
print(louvain)
for key in louvain.keys():
    for node in louvain[key]:
        print(node['name'] + " " + str(key))
"""

"""

for r, d, f in os.walk("experiment/test/"):
    if len(f) >  0:
        try: 
            project_name  = f[0][:-4]

            for row_idx in range(0, xl_sheet.nrows -2):    # Iterate through rows
                    gitname = xl_sheet.cell(row_idx, 0).value 
                    dataname = gitname.split("/")[-1]
                    if dataname == project_name:
                        print(gitname)
                        giturl = "https://github.com/" + gitname
                        etl = ETL(itype='git',iloc=giturl, token=None)
                        etl.run()
                        delete_filesystem = "MATCH p=()-[r:FILESYSTEM]->() DELETE p"
                        etl.graph_query(delete_filesystem)

                        louvain = etl.analytics("louvain")
                        print(louvain)
                        break



"""



            
"""
            statement = 'MATCH (n)-[r]->(m) RETURN r' # For now, we only uses edges
            ret = etl.graph_query(statement)['edges']
            print(ret)
            with open("experiment/train/"+dataname+"/"+dataname+".txt", 'w+') as f:

                for edge in ret:
                    if edge['type'] != 'FILESYSTEM':
                        f.write(str(edge['start_node']['id']) + " "+  str(edge['end_node']['id']) +" "+ str(edge['properties']['degree']) + "\n" ) 
"""

            #etl.clean_db() 
        #except Exception:
         #   pass
            #etl = ETL(itype='git',iloc=giturl, token=None)
            #etl.clean_db() 
            
            #Test DataSet
