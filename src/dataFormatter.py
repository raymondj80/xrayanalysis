import matplotlib.pyplot as plt
import nbformat as nbf
import os
import json

class Formatter:

    def plotSemilogy(self,x,y,x_min,x_max):
        plt.semilogy(x,y)
        plt.xlabel(r'Angle (2$\Theta$)')
        plt.ylabel(r'Intensity (Arb. Units)')
        plt.xlim([x_min,x_max])
        plt.ylim([1,1e7])

    def createJupyterNB(self,filename,foldername,dictionary):
        nb = nbf.v4.new_notebook()
        self.makeFolder(foldername)
        for k, v in dictionary.items():
            nb['cells'].extend([nbf.v4.new_markdown_cell(k), nbf.v4.new_code_cell(v) ])
        nbf.write(nb,'%s/%s.ipynb' % (foldername, filename))


    # create folder
    def makeFolder(self,foldername):
        if not os.path.exists(foldername):
            os.makedirs(foldername)

    # format Jupyter notebook
    def jupyterFormatter(self,raw_data,fit_data):
        raw_data_string = json.dumps(raw_data)
        fit_data_string = json.dumps(fit_data)
        json_dict = {
                """Import Dependencies""":
                """\
                import matplotlib.pyplot as plt
                """,
                """Load Data""":
                """\
                RAW_DATA = %s
                FIT_DATA = %s
                """ % (raw_data_string, fit_data_string),
                """Plot Data""":
                """\
                plt.semilogy(list(RAW_DATA.keys()),list(RAW_DATA.values()))
                plt.semilogy(list(FIT_DATA.keys()),list(FIT_DATA.values()))
                plt.xlabel(r'Angle (2$\Theta$)')
                plt.ylabel(r'Intensity (Arb. Units)')
                plt.xlim([400,800])
                plt.ylim([1,1e7])
                """
            }
        return json_dict

