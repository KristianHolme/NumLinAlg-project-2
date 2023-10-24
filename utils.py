import testing

def precompAll():
    #run all jitted functions on small problems to compile them
    testing.runLanczos(N=2, k=2, usejit=True, orth=True, verbose=False)
    

    