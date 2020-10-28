import rpy2.robjects as ro

pi = ro.r('pi')
print(pi[0])

ro.r(''' 
    f <- function(r, verbose=FALSE) {
            if (verbose) { cat("I am calling f().\n") }
                2 * pi * r
            }
            f(3) 
     ''')


rf = ro.globalenv['f']
print(rf.r_repr())
 
res = rf(3)
print(res)

r = ro.r

x = ro.IntVector(range(10))
y = r.rnorm(10)

r.X11()
r.layout(r.matrix(ro.IntVector([1,2,3,2]), nrow=2, ncol=2))
r.plot(r.runif(10), y, xlab="runif", ylab="foo/bar", col="red")
input("Press Enter to continue...")


