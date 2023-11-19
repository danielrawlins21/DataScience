import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")
K = [1,2,3,4]


#Kmeans parameters init
costs_Kmeans = [0,0,0,0,0]
best_seed_Kmeans = [0,0,0,0,0]
mixtures_Kmeans = [0,0,0,0,0]
posts_Kmeans = [0,0,0,0,0]

#Naive Bayes parameters init
costs_EM = [0,0,0,0,0]
best_seed_EM = [0,0,0,0,0]
mixtures_EM = [0,0,0,0,0]
posts_EM = [0,0,0,0,0]

bic = [0.,0.,0.,0.]
for k in K:
    for seed in [0,1,2,3,4]:
        init_model = common.init(X,k,seed)
        #breakpoint()
        mixtures_Kmeans[seed],posts_Kmeans[seed],costs_Kmeans[seed] = kmeans.run(X,*init_model)

        #Naive bayes EM
        mixtures_EM[seed], posts_EM[seed],costs_EM[seed] = naive_em.run(X,*init_model)
    #print("|||||||| Clusters-",k," ||||||||")
    #print("Lowest cost using Kmeans is:",np.min(costs_Kmeans))
    #print("Highest cost using EM is:",np.max(costs_EM))

    #print()
    #best_seed_Kmeans[k] = np.armin(costs_Kmeans)
    """common.plot(X,
                mixtures_Kmeans[best_seed_Kmeans[k]],
                posts_Kmeans[best_seed_Kmeans[k]],
                title="Kmeans")
    common.plot(X,
                mixtures_EM[best_seed_EM[k]],
                posts_EM[best_seed_EM[k]],
                title="EM")   """
    bic[k-1] = common.bic(X,mixtures_EM[best_seed_EM[k-1]],np.max(costs_EM))

print("================= BIC ====================")
print("Best K is:", np.argmax(bic)+1)
print("BIC for the best K is:", np.max(bic))
