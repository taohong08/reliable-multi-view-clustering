%%%% demo_RMVC
load data
nCluster = length(unique(gnd));

[y0, alpha,results, findY0_time,qp_time, findY_time]= RMVC(Yi,Y0,nCluster,gnd);