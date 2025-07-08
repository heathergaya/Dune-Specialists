# library(terra)
# library(tidyterra)
# library(sf)
# library(nimbleSCR)
# surveys<-read.csv("Ipswich site visits.csv", header=T)
# effort <- read.csv('detail_effort.csv')
# effort_full <- subset(effort, effort$Full.Partial == "F")
# habitat <- vect('habitat_update.shp')
# habitat <- project(habitat, 'EPSG:26918')
# bird_dets<-vect("ipswich_points.shp")
# bird_dets <- project(bird_dets, 'EPSG:26918')
# transects <- vect('wiggly_merge.shp')
# transects <- project(transects, 'EPSG:26918')
# 
# partialsurvs <- vect('partial_surveys_study_area/partial_surveys_study_area.shp')
# partialsurvs <- project(partialsurvs, 'EPSG:26918')
# effort_partial <- subset(effort, effort$Full.Partial == "P")
# 
# veg <- rast('nummy_veg.tif')
# setwd("~/Desktop/U_Georgia/Sparrows/Dune Specialists/StoneHarbor_Orthomosaic_export_SatDec01231434.tif")
# veg2 <- rast('StoneHarbor_Orthomosaic_export_SatDec01231434.tif')
# veg2 <- project(veg2, 'EPSG:26918')
# water <- vect('water.shp')
# water <- project(water, 'EPSG:26918')
# #myveg <- terra::extract(veg2, bird_dets, bind = TRUE)
# setwd("~/Desktop/U_Georgia/Sparrows/Dune Specialists")
# 
# goodhab <- veg
# goodhab[goodhab$nummy_veg %in% c(3:9)] <- 0
# goodhab[goodhab >0] <- 1
# plot(goodhab)
# # plot(beep)
# # beep2 <- beep
# # beep2[beep2$Zone5_DEM > 2.01] <- 0
# # beep2[beep2$Zone5_DEM >0] <- 1
# # plot(beep2)
# # plot(veg2$StoneHarbor_Orthomosaic_export_SatDec01231434_1)
# # myveg2 <- terra::extract(beep, myveg, bind = TRUE)
# # xseq<- seq(518800, 519800, by = 10)
# # yseq <- seq(4319800, 4320600, by =10)
# # pts <- expand.grid(xseq, yseq)
# # v1 <- terra::extract(veg2$StoneHarbor_Orthomosaic_export_SatDec01231434_1, pts, bind = TRUE)
# # v2 <- terra::extract(beep2, v1, bind = T)
# # v2 <- subset(v2, !is.na(v2$StoneHarbor_Orthomosaic_export_SatDec01231434_1))
# # plot(v2)
# #plot(v2$StoneHarbor_Orthomosaic_export_SatDec01231434_1, v2$Zone5_DEM)
# 
# F.survs1<-surveys[surveys$F.Full.P.Partial=="F" & surveys$RECORD %in% effort_full$RECORD,] # Joe was able to designate a list of full surveys
# F.survs1 <- F.survs1[order(F.survs1$julian),]
# table(F.survs1$DATE, F.survs1$SITE)
# table(effort_partial$DATE, effort_partial$SECTOR)
# nocc <- length(unique(c(F.survs1$julian, effort_partial$julian)))
# 
# ext(habitat)
# #turn habitat into coarse raster
# rasty <- rast(nrows = 30, ncols = 30, ext(habitat))
# res(rasty) <- c(50, 50)
# bounds <- rasterize(habitat, rasty, field = 'habitat', touches = T)
# plot(bounds)
# plot(transects, add = T, col = 'white')
# 
# ### Time to figure out what proportion of each pixel is suitable
# prop_good1 <- resample(goodhab, bounds)
# rasty2 <- rast(nrows = 300, ncols = 300, ext(habitat))
# res(rasty2) <- c(5, 5)
# watery <- rasterize(water, rasty2, field = 'id', touches = T)
# watery[!is.na(watery)] <- 0
# watery[is.na(watery)] <- 1
# watery2 <- resample(watery, bounds)
# watery2 <- mask(watery2, bounds)
# watery2 <- watery2 %>%
#   filter(
#     x > 518600
#   )
# crs(bounds) <- crs(watery2) <- crs(prop_good1)
# propgood1 <- mask(prop_good1, bounds)
# propgood2 <- mask(watery2, bounds)
# add(propgood1) <- propgood2
# veg_layer <- sum(propgood1, na.rm = T)
# plot(veg_layer)
# 
# effort_rast <- rasterizeGeom(transects, rasty, "length")
# 
# ## We will do an unmarked SCR analysis
# # First we need to get the two raster layers prepared. First is habitat type:
# hab_rast_pre <- c(t(as.matrix(veg_layer, wide = T)))
# hab_rast <- hab_rast_pre[!is.na(hab_rast_pre)]
# 
# eff_mask <- c(t(as.matrix(effort_rast, wide = TRUE)))
# eff_mask <- eff_mask[!is.na(hab_rast_pre)] #turn into matrix and remove unwanted pixels
# eff_mask_m <- mean(eff_mask)
# eff_mask_sd <- sd(eff_mask)
# eff_mask_s <- (eff_mask - eff_mask_m)/eff_mask_sd #scale
# 
# trap_locs <- as.data.frame(bounds, xy = T)
# plot(veg_layer)
# points(trap_locs[,1:2], pch = 19, cex = .5)
# 
# coordsObsCenter <- trap_locs[,1:2]     # Extract x and y coordinates from trap locations
# 
# # Rescale coordinates to match the habitat grid
# # this requires the points are in UTMS
# scaledObjects <- scaleCoordsToHabitatGrid(        # Re-scaling the entire coordinate system of the data input is a requirement to run SCR models with the local evaluation approach.
#   coordsData = coordsObsCenter,                   # Coordinates of observed hair traps
#   coordsHabitatGridCenter = crds(bounds)  # Habitat grid center coordinates
# )
# 
# # Get lower and upper coordinates of each square cell (flips raster upside down but that's normal)
# 
# lowerAndUpperCoords <- getWindowCoords(
#   scaledHabGridCenter = scaledObjects$coordsHabitatGridCenterScaled,   # Scaled habitat grid
#   scaledObsGridCenter = scaledObjects$coordsDataScaled,                # Scaled observation data
#   plot.check = TRUE                                                    # Check the plot of the window coordinates
# )
# 
# # Set maximum distance for local object analysis
# # Identify all objects (e.g. traps) within a given radius dmax of each cell in a habitat mask
# # The distance to the activity center and the detection probability are then calculated for local objects only
# # (i.e. the detection probability is assumed to be 0 for all other objects as they are far enough from the activity center).
# 
# # Convert the raster to a matrix for analysis
# habitatMask <- as.matrix(bounds, wide = TRUE)
# habitatMask[!is.na(habitatMask)] <- 1
# 
# trapLocal <- getLocalObjects(habitatMask = habitatMask,                # Local objects based on habitat mask
#                              coords = scaledObjects$coordsDataScaled,  # Scaled observation coordinates
#                              dmax = 3,                                # Maximum distance; should be as small as possible in order to reduce computation; will be smaller with larger pixels sizes but larger with larger resize factors
#                              resizeFactor = 1,                         # Aggregation factor; aggregate habitat cells for lower resolution, obtain objects with smaller dimensions
#                              plot.check = TRUE)                        # Displays which traps are considered "local" for a randomly chosen habitat cell.
# 
# dim(trapLocal$habitatGrid) == dim(lowerAndUpperCoords$habitatGrid) #must be true
# 
# # Display a table of local indices, count total number traps
# table(unname(trapLocal$numLocalIndices))  # Count the number of local indices
# ntraps <- nrow(trap_locs)                 # Count the number of traps
# ntraps
# 
# ## Get detections in order:
# myoccasions <- sort(unique(c(F.survs1$julian, effort_partial$julian)))
# bird.pts <- as.data.frame(bird_dets, geom="XY")[,c(1:10, 45:46)]
# bird.pts$j_site <- paste0(bird.pts$julian-2000000, bird.pts$SITE)
# F.survs1$j_site <- paste0(F.survs1$julian, F.survs1$SITE)
# effort_partial$j_site <- paste0(effort_partial$julian, effort_partial$SITE)
# bird.pts <- subset(bird.pts, bird.pts$j_site %in% unique(c(F.survs1$j_site,effort_partial$j_site)))
# bird.pts$j2 <- bird.pts$julian-2000000
# bird.pts$cells <- cellFromXY(bounds, bird.pts[,c('x','y')])
# trap_locs$cells <- cellFromXY(bounds, trap_locs[,c('x','y')])
# trap_locs$id <- 1:nrow(trap_locs)
# bird.pts$trap <- trap_locs[match(bird.pts$cells, trap_locs$cells), 'id']
# 
# F.survs2 <- rbind(F.survs1[,-6], effort_partial[,-c(6:7)])
# F.survs2$survey_f <- factor(F.survs2$julian, levels = myoccasions)
# F.survs2$Primary <- ifelse(F.survs2$julian < 17351, 1,
#                            ifelse(F.survs2$julian < 18001, 2,
#                                   ifelse(F.survs2$julian < 18014, 3,
#                                          ifelse(F.survs2$julian <= 18028, 4,
#                                                 ifelse(F.survs2$julian <= 18042, 5,
#                                                        ifelse(F.survs2$julian <= 18056, 6,
#                                                               ifelse(F.survs2$julian <= 18070, 7,8
#                                                               )))))))
# 
# F.survs1$Primary <- ifelse(F.survs1$julian < 17351, 1,
#                            ifelse(F.survs1$julian < 18001, 2,
#                                   ifelse(F.survs1$julian < 18014, 3,
#                                          ifelse(F.survs1$julian <= 18028, 4,
#                                                 ifelse(F.survs1$julian <= 18042, 5,
#                                                        ifelse(F.survs1$julian <= 18056, 6,
#                                                               ifelse(F.survs1$julian <= 18070, 7,8
#                                                               )))))))
# nPrimary <- 8 #2 fortnights are in december
# library(dplyr)
# F.survs2 <- F.survs2 %>%
#   mutate(secondary = ave(Primary, Primary, FUN = seq_along))
# 
# dets <- array(0, c(ntraps, nPrimary,max(F.survs2$secondary)))
# for(i in 1:nrow(bird.pts)){
#   o <- which((F.survs2$survey_f) == bird.pts$j2[i])
#   p <- F.survs2[o, 'Primary']
#   s <- F.survs2[o, 'secondary']
#   tt <- bird.pts$trap[i]
#   dets[tt,p,s] <- 1
# }
# 
# n_sum <- apply(dets, c(1, 2), sum)
# 
# #Now we have to quantify effort
# # How many secondary periods was each trap surveyed
# F.survs1$Primary <- factor(F.survs1$Primary, levels = as.character(1:nPrimary))
# ff <- table(F.survs1$Primary, F.survs1$SITE)
# effort <- array(0, c(ntraps, nPrimary)) #trials
# nn <- which(trap_locs$habitat == 'Nummy')
# sh <- which(trap_locs$habitat != 'Nummy')
# LM <- which(trap_locs$habitat == 'Low marsh')
# TT <- which(trap_locs$habitat == 'Transitional')
# DD <- which(trap_locs$habitat == 'Dune')
# for(t in 1:nPrimary){
#   effort[nn,t] <- ff[t,1]
#   effort[sh,t] <- ff[t,2]
# }
# 
# ### Got to add in the partial survey times:
# partialguys <- subset(F.survs2, F.survs2$RECORD %in% effort_partial$RECORD)
# partialguys$Primary <- factor(partialguys$Primary, levels = as.character(1:nPrimary))
# ff2 <- table(partialguys$Primary, partialguys$SECTOR)
# dunes1 <- relate(partialsurvs[1], vect(as.matrix(trap_locs[,c('x', 'y')])), "contains") |> which()
# dunes1 <- c(dunes1, c(231,294))
# rearsouthmarsh <- relate(partialsurvs[2], vect(as.matrix(trap_locs[,c('x', 'y')])), "contains") |> which()
# frontdune <- relate(partialsurvs[3], vect(as.matrix(trap_locs[,c('x', 'y')])), "contains") |> which()
# rear <- relate(partialsurvs[4], vect(as.matrix(trap_locs[,c('x', 'y')])), "contains") |> which()
# rear <- c(rear, 293)
# for(t in 1:nPrimary){
#   effort[c(dunes1,rear), t] <- effort[c(dunes1, rear),t]+ ff2[t,1]
#   effort[frontdune,t] <- effort[frontdune,t]+ ff2[t,2]
#   effort[frontdune,t] <- effort[frontdune,t] +ff2[t,3]
#   effort[c(frontdune, rear),t] <- effort[c(frontdune, rear),t] + ff2[t,4]
#   effort[rear,t] <- effort[rear,t] + ff2[t,5]
#   effort[rearsouthmarsh,t] <- effort[rearsouthmarsh,t]+ ff2[t,6]
# }
# 
# 
# #some pixels were never surveyed:
# `%notin%` <- Negate(`%in%`)
# nosurv <- which(eff_mask == 0)
# nosurv_corrected <- nosurv[nosurv %notin% unique(bird.pts$trap)]
# adjust <- nosurv_corrected[nosurv_corrected %in% unique(bird.pts$trap)]
# effort[nosurv_corrected,] <- 0
# #effort[adjust,] <- 1
# #effort[c(234,235,183), 1] <- 1 #oops
# effort[c(216,241, 256, 268, 279), 2] <- 10 #oops 
# n_sum[c(294, 250),2] <- 0 #weren't surveyed by a full survey 
# effort[c(61,66,69,111,139),2] <- 2
# effort[293,3] <- 0 #oops again 
# n_sum[c(293,294, 286),3] <- 0
# effort[c(296, 279, 268,102,216,237),3] <- 4
# effort[c(265,266),3] <- 1
# 
# 
# hab.trap <- array(NA, ntraps)
# hab.trap[nn] <- 3
# hab.trap[LM] <- 2
# hab.trap[DD] <- 1
# hab.trap[TT] <- 4
# 
library(nimbleSCR)
get_p <- nimbleFunction(
  run = function(s = double(2),
                 M = double(0),
                 lam0=double(1),
                 sigma=double(1),
                 trapCoords = double(2),
                 ntraps=double(0),
                 z = double(1),
                 ntrials = double(1),
                 localTrapsNum = double(1),
                 localTrapsIndices = double(2),
                 habitatGrid = double(2),
                 trap.hab = double(1)){
    returnType(double(1))

    sID <- array(NA, M)
    lam <- array(0, c(M, ntraps))
    Lambda <- array(0, ntraps)
    p <- array(0, ntraps)

    for(i in 1:M){
     sID[i] <- habitatGrid[trunc(s[i,2])+1, trunc(s[i,1])+1]
     }

    for(j in 1:ntraps){
      lam0s <- lam0[trap.hab[j]]
      if(ntrials[j] == 0){ #if closed:
        p[j] <- 0
      } else{ #if open:
      for(i in 1:M){
        if(z[i] > 0){#if not real, p = 0
          mynum <- localTrapsNum[sID[i]]
          theseLocalTraps <- localTrapsIndices[sID[i], 1:mynum]
          if(sum(theseLocalTraps == j) >0) {
          d2 <- sqrt((s[i,1]-trapCoords[j,1])^2 + (s[i,2]-trapCoords[j,2])^2)
          lam[i,j] <- lam0s*exp(-d2^2/(2*sigma[j]^2))
          }
        }
      } #end i
      Lambda[j] <- sum(lam[1:M,j])
      p[j] <- 1 - exp(-Lambda[j])
    } #end open traps
      }#end j

    return(p)
  }
)

Sparrows <- nimbleCode({
  for(m in 1:numHabWindows){
    habIntensity[m] <- exp(beta0 + beta1*veg[m])*pixArea
    logHabIntensity[m] <- log(habIntensity[m])
  }

  sumHabIntensity <- sum(habIntensity[1:numHabWindows]) #total expected sparrows
  logSumHabIntensity <- log(sumHabIntensity)

  beta0 ~ dnorm(0,1) #intercept
  beta1 ~ dnorm(0,1) #slope

  sig0 ~ dnorm(0, 1) #no idea
  sig1 ~ dgamma(1, 1) #more effort increases detection
  for(j in 1:n.traps) {
  sigma[j] <- exp(sig0 + sig1*eff[j]) #transect distance in pixel influences detection
  }

  #for(k in 1:4){ #habitat types
  #  lam0[k] ~ dbeta(1,1)
  #}
  lam0[1] ~ dbeta(1,1) #dune island
  lam0[2] <- lam0[1] #dune island
  lam0[3] ~ dbeta(1,1) #Nummy
  lam0[4] <- lam0[1] #dune island

  for(t in 1:nPrimary){
    psi[t] ~ dunif(0, 1)
    for(i in 1:M) { # loop over all sparrows
      z[i,t] ~ dbern(psi[t])

      #get an activity center regardless if real/fake
      s[i,1:2,t] ~ dbernppAC(
        lowerCoords = habLoCoords[1:numHabWindows,1:2],
        upperCoords = habUpCoords[1:numHabWindows,1:2],
        logIntensities = logHabIntensity[1:numHabWindows],
        logSumIntensity = logSumHabIntensity,
        habitatGrid = habitatGrid[1:y.max,1:x.max],
        numGridRows = y.max,
        numGridCols = x.max)
    } #end i

    p[1:n.traps,t] <- get_p( #get p(at least one det) summed across all individuals
      s = s[1:M, 1:2,t],
      M = M,
      lam0 = lam0[1:4],
      sigma = sigma[1:n.traps],
      trapCoords = trapCoords[1:n.traps,1:2],
      ntraps = n.traps,
      z = z[1:M,t],
      ntrials = trials[1:n.traps,t],
      localTrapsNum = nTraps[1:n.cells],
      localTrapsIndices = trapIndex[1:n.cells,1:maxNBDets],
      habitatGrid = habitatIDDet[1:y.maxDet,1:x.maxDet],
      trap.hab = trap.hab[1:n.traps])

    dens[1:all_cells, t] <- calculateDensity(s = s[1:M,1:2, t],
                                           habitatGrid =  habitatGrid[1:y.max,1:x.max],
                                           indicator = z[1:M, t],
                                           numWindows = all_cells,
                                           nIndividuals = M)

    for(j in 1:n.traps) {
      nsum[j,t] ~ dbin(p = p[j,t], size = trials[j,t])
    } #end j

    N[t] <- sum(z[1:M,t])
  } #end t
})
# 
# 
# nimdat <- list(trapCoords = scaledObjects$coordsDataScaled,
#                trials = effort,
#                nsum = n_sum,
#                habLoCoords = lowerAndUpperCoords$lowerHabCoords,
#                habUpCoords = lowerAndUpperCoords$upperHabCoords,
#                habitatGrid = lowerAndUpperCoords$habitatGrid)
# 
# 
# 
# nimconst <- list(nPrimary = nPrimary,
#                  M = 400,
#                  pixArea = prod(res(veg_layer)),
#                  veg = hab_rast,
#                  eff = eff_mask_s,
#                  n.traps = dim(scaledObjects$coordsDataScaled)[1],
#                  y.max = dim(habitatMask)[1],
#                  x.max = dim(habitatMask)[2],
#                  y.maxDet = dim(trapLocal$habitatGrid)[1],
#                  x.maxDet = dim(trapLocal$habitatGrid)[2],
#                  resizeFactor = trapLocal$resizeFactor,
#                  n.cells = dim(trapLocal$localIndices)[1],
#                  maxNBDets = trapLocal$numLocalIndicesMax,
#                  trapIndex = trapLocal$localIndices,
#                  nTraps = trapLocal$numLocalIndices,
#                  habitatIDDet = trapLocal$habitatGrid,
#                  numHabWindows = dim(lowerAndUpperCoords$lowerHabCoords)[1],
#                  all_cells = prod(dim(lowerAndUpperCoords$habitatGrid)),
#                  trap.hab = hab.trap)
# set.seed(1)
# M <- 600
# init.z <- array(rbinom(M*nPrimary, 1, .9), c(M, nPrimary))
# init.z[1:ntraps,] <- 1
# init.s <- array(NA, c(M, 2, nPrimary))
# iii <- sample(1:ntraps, length((ntraps+1):M), replace = T)
# for(t in 1:nPrimary){
# init.s[1:ntraps,,t] <- as.matrix(scaledObjects$coordsDataScaled)+.1
# init.s[(ntraps+1):M,1:2,t] <- as.matrix(scaledObjects$coordsDataScaled[iii,])
# for(i in 1:M){
#   success <- F
#   if(trapLocal$habitatGrid[trunc(init.s[i,2,t])+1, trunc(init.s[i,1,t])+1] == 0){
#     while (!success) {
#     init.s[i,,t] <- as.matrix(scaledObjects$coordsDataScaled[sample(1:ntraps, 1),])
#     if(trapLocal$habitatGrid[trunc(init.s[i,2,t])+1, trunc(init.s[i,1,t])+1] >0 ){
#       success <- T
#     }
#     }
#   }
# }
# init.s[M,2,] <- 5
# init.s[M,1,] <- 13
# }
# 
# niminits <- list(beta0 = 2,
#                  beta1 = .5,
#                  z = init.z,
#                  s = init.s,
#                  psi = rep(.9, nPrimary),
#                  lam0 = rep(.3, 4),
#                  sig0 = .1,
#                  sig1 = 0)
# 
# nimstuff <- list(inits = niminits, consts = nimconst, dat = nimdat)
# saveRDS(nimstuff, 'nimstuff2.rds')
nimstuff <- readRDS('nimstuff2.rds')
niminits <- nimstuff$inits
nimdat <- nimstuff$dat
nimconst <- nimstuff$consts

params <- c('beta0', 'lam0', 'psi', 'N', 'sig1', 'sig0', 'dens', 'beta1')
library(parallel)
cl <- makeCluster(3)
clusterExport(cl = cl, varlist = c('niminits', "params", "nimconst", 'nimdat', 'Sparrows', 'get_p'))
system.time(
  nim.out <- clusterEvalQ(cl = cl,{
    library(nimble)
    library(coda)
    library(nimbleSCR)
    source('runMCMCbites.R')
    prepnim <- nimbleModel(code = Sparrows, constants = nimconst,
                           data = nimdat, inits = niminits, calculate = T)
    
    prepnim$calculate() #if this is NA or -Inf you know it's gone wrong
    mcmcnim <- configureMCMC(prepnim, monitors = params, print = T)
    
    nimMCMC <- buildMCMC(mcmcnim) #actually build the code for those samplers
    
    #Cmodel <- compileNimble(prepnim) #compiling the model itself in C++;
    #
    #Compnim <- compileNimble(nimMCMC, project = prepnim) # compile the samplers next
    #Compnim$run(niter = 50000, nburnin = 30000, thin = 1)
    #return(as.mcmc(as.matrix(Compnim$mvSamples)))
    Rmcmc <- buildMCMC(mcmcnim) #actually build the code for those samplers
    
    Compnim <- compileNimble(list(model = prepnim,
                                  mcmc = Rmcmc),
                             showCompilerOutput = F)
    Cmcmc <- Compnim$mcmc
    Cmodel <- compileNimble(prepnim) #compiling the model itself in C++;
    c <- round(runif(1),4)
    runMCMCbites( mcmc = Cmcmc,                           ## Compiled MCMC
                  model = Cmodel,                         ## Compiled model code
                  conf = mcmcnim,                            ## MCMC configuration
                  bite.size = 500,                        ## Number of iterations per bite
                  bite.number = 20,                        ## Number of MCMC bites
                  path = paste0("SparrowApril23/chain",c),          ## Directory where MCMC outputs will be saved
                  save.rds = TRUE)                        ## Option to save the state of the model 
    
    nimOutput <- collectMCMCbites(path = paste0("SparrowApril23/chain",c),           ## Directory containing MCMC outputs 
                                  burnin = 5,              ## Number of MCMC bites to ignore as burnin 
                                  pattern = "mcmcSamples",
                                  param.omit = NULL,
                                  progress.bar = T)
    
    return(nimOutput)
    
  }) #this will take awhile and not produce any noticeable output.
) #Slow! 

library(coda)
nim.out <- as.mcmc.list(nim.out)
saveRDS(nim.out, 'sparrows_outApril23.rds')
# 
# ### Output:
# library(coda)
# library(ggplot2)
# library(MCMCvis)
# nim.out <- readRDS('Mar11out.RDS')
# 
# #N all
# 
# MCMCsummary(nim.out, params = 'N')
# 
# ds <- summary(nim.out[,paste0('dens[', 1:840, ", ", rep(1:6, each = 840), ']'),])$statistics
# s1 <- s2 <- s3 <- s4 <- s5 <- s6 <- bounds
# s1[] <- s2[] <- s3[] <- s4[] <- s5[] <- s6[] <- NA
# good <- which(c(t(nimdat$habitatGrid)) != 0)
# s1[good] <- ds[1:297,1]
# s2[good] <- ds[(1:297)+840,1]
# s3[good] <- ds[(1:297)+(840)*2,1]
# s4[good] <- ds[(1:297)+(840)*3,1]
# s5[good] <- ds[(1:297)+(840)*4,1]
# s6[good] <- ds[(1:297)+(840)*5,1]
# par(mfrow = c(3,2))
# plot(s1)
# plot(s2)
# plot(s3)
# plot(s4)
# plot(s5)
# plot(s6)
