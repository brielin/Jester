# pylmm is a python-based linear mixed-model solver with applications to GWAS

# Copyright (C) 2013  Nicholas A. Furlotte (nick.furlotte@gmail.com)

#The program is free for academic use. Please contact Nick Furlotte
#<nick.furlotte@gmail.com> if you are interested in using the software for
#commercial purposes.

#The software must not be modified and distributed without prior
#permission of the author.

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
#CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
import numpy as np
import struct
import pdb
from IPython import embed

class plink:
   def __init__(self,fbase,type='b',kFile=None,phenoFile=None,sampleFile=None,
                covFile=None,normGenotype=True,remMissing=False,readKFile=False,
                fastLMM_kinship=False,noMean=False):
      self.fbase = fbase
      self.type = type
      if not ((type == 'emma') | (type == 'g') | (type == 's')):
         self.indivs = self.getIndivs(self.fbase,sampleFile,type)
      elif type == 'emma':
         #Just read a line from the SNP file and see how many people we have
         f = open(fbase,'r')
         self.N = len(f.readline().strip().split())
         self.indivs = [(str(x),str(x)) for x in range(1,self.N+1)]
         f.close()
      elif type == 'g':
         f = open(fbase+'.geno','r')
         self.N = len(f.readline()[:-1])
         self.indivs = [(str(x),str(x)) for x in range(1,self.N+1)]
         f.close()
      elif type == 's':
         self.N = 0
         self.indivs = []

      self.kFile = kFile
      self.phenos = None
      self.normGenotype = normGenotype
      self.remMissing = remMissing
      self.phenoFile = phenoFile
      self.covFile = covFile
      # Originally I was using the fastLMM style that has indiv IDs embedded.
      # NOW I want to use this module to just read SNPs so I'm allowing
      # the programmer to turn off the kinship reading.
      self.readKFile = readKFile
      self.fastLMM_kinship_style = fastLMM_kinship

      if self.kFile: self.K = self.readKinship(self.kFile)
      elif os.path.isfile("%s.kin" % fbase):
         self.kFile = "%s.kin" %fbase
         if self.readKFile: self.K = self.readKinship(self.kFile)
      else:
         self.kFile = None
         self.K = None

      self.getPhenos(self.phenoFile)
      self.getCovariates(self.covFile,noMean)
      self.fhandle = None
      self.snpFileHandle = None

   def __del__(self):
      if self.fhandle: self.fhandle.close()
      if self.snpFileHandle: self.snpFileHandle.close()

   def getSNPIterator(self):
      if self.type == 'b': return self.getSNPIterator_bed()
      elif self.type == 't': return self.getSNPIterator_tped()
      elif self.type == 'emma': return self.getSNPIterator_emma()
      elif self.type == 'g': return self.getSNPIterator_geno()
      elif self.type == 'i': return self.getSNPIterator_gens()
      elif self.type == 's': return self.getSNPIterator_summary()
      else:
         sys.stderr.write("Please set type to either b or t\n")
         return

   def getSNPIterator_emma(self):
      # get the number of snps
      file = self.fbase + '.snp'
      if not os.path.isfile(file): file = self.fbase + '.map'
      if not os.path.isfile(file):
         sys.stderr.write("No SNP or MAP file" + file + " found. Exiting\n")
         exit(1)
      i = 0
      f = open(file,'r')
      for line in f: i += 1
      f.close()
      self.numSNPs = i
      self.have_read = 0
      self.snpFileHandle = open(file,'r')
      file = self.fbase
      self.fhandle = open(file,'r')

      return self

   def getSNPIterator_geno(self):
      # get the number of snps
      file = self.fbase + '.snp'
      if not os.path.isfile(file): file = self.fbase + '.map'
      if not os.path.isfile(file):
         sys.stderr.write("No SNP or MAP file" + file + " found. Exiting\n")
         exit(1)
      f = open(file,'r')
      n = sum( 1 for line in f )
      f.close()
      self.numSNPs = n
      self.have_read = 0
      self.snpFileHandle = open(file,'r')
      file = self.fbase + '.geno'
      self.fhandle = open(file,'r')

      return self

   def getSNPIterator_gens(self):
      self.have_read = 0
      #get the number of snps
      file = self.fbase + '.I2_info'
      f = open(file, 'r')
      n = sum( 1 for line in f ) - 1
      f.close()
      self.numSNPs = n
      self.have_read = 0
      self.snpFileHandle = open(file,'r')
      file = self.fbase + '.I2'
      self.fhandle = open(file,'r')
      return self

   def getSNPIterator_tped(self):
      # get the number of snps
      file = self.fbase + '.bim'
      if not os.path.isfile(file): file = self.fbase + '.map'
      i = 0
      f = open(file,'r')
      for line in f: i += 1
      f.close()
      self.numSNPs = i
      self.have_read = 0
      self.snpFileHandle = open(file,'r')

      file = self.fbase + '.tped'
      self.fhandle = open(file,'r')

      return self

   def getSNPIterator_bed(self):
      # get the number of snps
      file = self.fbase + '.bim'
      i = 0
      f = open(file,'r')
      for line in f: i += 1
      f.close()
      self.numSNPs = i
      self.have_read = 0
      self.snpFileHandle = open(file,'r')

      self.BytestoRead = self.N / 4 + (self.N % 4 and 1 or 0)
      self._formatStr = 'c'*self.BytestoRead

      file = self.fbase + '.bed'
      self.fhandle = open(file,'rb')

      magicNumber = self.fhandle.read(2)
      order = self.fhandle.read(1)
      if not order == '\x01':
         sys.stderr.write("This is not in SNP major order\n")
         raise StopIteration
      return self

   def getSNPIterator_summary(self):
      file = self.fbase + '.txt' #Might need to change this
      self.numSNPs = sum( 1 for line in open(file,'r'))
      self.have_read = 0
      self.snpFileHandle = open(file,'r')
      self.fhandle = open(file,'r')
      return self

   def __iter__(self): return self.getSNPIterator()

   def next(self):
      if self.have_read == self.numSNPs: raise StopIteration
      self.have_read += 1

      if self.type == 'b':
         X = self.fhandle.read(self.BytestoRead)
         XX = [bin(ord(x)) for x in struct.unpack(self._formatStr,X)]
         bimLine = self.snpFileHandle.readline().strip().split()
         return (self.formatBinaryGenotypes(XX), bimLine[0],
                 bimLine[1], int(bimLine[3]))

      elif self.type == 't':
         X = self.fhandle.readline()
         XX = X.strip().split()
         chrm,rsid,pos1,pos2 = tuple(XX[:4])
         XX = XX[4:]
         G = self.getGenos_tped(XX)
         if not self.remMissing: G = self.replaceMissing(G)
         if self.normGenotype: G = self.normalizeGenotype(G)
         return G,chrm,rsid,int(pos2)

      elif self.type == 'emma':
         X = self.fhandle.readline()
         if X == '': raise StopIteration
         XX = X.strip().split()
         G = []
         for x in XX:
            try:
               G.append(float(x))
            except: G.append(np.nan)
         G = np.array(G)
         if not self.remMissing: G = self.replaceMissing(G)
         if self.normGenotype: G = self.normalizeGenotype(G)
         return G,0,self.snpFileHandle.readline().strip().split()[0], 0

      elif self.type == 'g':
         X = self.fhandle.readline()
         if X == '': raise StopIteration
         #G = np.array( [ float(x) for x in X[:-1] ] )
         def f(x):
            if (float(x) >= 0) & (float(x) <= 2): return float(x)
            else: return np.nan
         G = np.array( map( f(x),  X[:-1] )) ## A bit slow but same asloop :(
         if not self.remMissing: G = self.replaceMissing(G)
         if self.normGenotype: G = self.normalizeGenotype(G)
         return G, 0, self.snpFileHandle.readline().strip().split()[0], 0

      elif self.type == 'i':
         X = self.fhandle.readline()
         if X == '': raise StopIteration
         X = X.strip().split()
         chrm = X[0]
         rsid = X[1]
         pos = X[2]
         def f(x,y):
            return 0.5*float(x) + float(y)
         ##Compute dosages based on genotype probabilities, X[0:4] are snp info.
         G = np.array( map( f, X[6::3], X[7::3] ) )
         if not len(G) == self.N:
            print "SNPs read does not match number of individuals. Exiting"
            sys.exit(1)
         #No missing genotypes in imputed files, but could be for general .gens
         if not self.remMissing: G = self.replaceMissing(G)
         if self.normGenotype: G = self.normalizeGenotype(G)
         return G, chrm, rsid, int(pos)

      elif self.type == 's':
         X = self.fhandle.readline()
         if X == '': raise StopIteration
         (chrm,rsid,pos,a1,a2,maf,ncase,ncont,beta,betaSE,pV)=X.strip().split()
         return int(chrm),rsid,int(pos),float(maf),a1,a2,int(ncase),int(ncont),\
             float(beta),float(betaSE),float(pV)

      else: sys.stderr.write("Do not understand type %s\n" % (self.type))

   def getGenos_tped(self,X):
      G = []
      for i in range(0,len(X)-1,2):
         a = X[i]
         b = X[i+1]
         if a == b == '0': g = np.nan
         if a == b == '1': g = 0
         if a == b == '2': g = 1
         if a != b: g = 0.5
         G.append(g)
      return np.array(G)

   def formatBinaryGenotypes(self,X):
      D = { \
         '00': 0.0, \
         '10': 0.5, \
         '11': 1.0, \
         '01': np.nan \
         }

      G = []
      for x in X:
         if not len(x) == 10:
            xx = x[2:]
            x = '0b' + '0'*(8 - len(xx)) + xx
         a,b,c,d = (x[8:],x[6:8],x[4:6],x[2:4])
         L = [D[y] for y in [a,b,c,d]]
         G += L
         # only take the leading values because whatever is left should be null
      G = G[:self.N]
      G = np.array(G)
      if not self.remMissing: G = self.replaceMissing(G)
      if self.normGenotype: G = self.normalizeGenotype(G)
      return G

   def replaceMissing(self,G):
      x = True - np.isnan(G)
      if not len(G[x]): return G[x]
      m = G[x].mean()
      G[np.isnan(G)] = m
      return G

   ## This used to normalize the genotypes and replace missing values with MAF.
   ## Now JUST performs normalization. Missing value handling done elsewhere.
   def normalizeGenotype(self,G):
      x = True - np.isnan(G)
      if not len(G[x]): return G[x]
      m = G[x].mean()
      if G[x].var() == 0: s = 1.0
      else: s = np.sqrt(G[x].var())
      if s == 0: G = G - m
      else: G = (G - m) / s
      return G

   def getPhenos(self,phenoFile=None):
      if not phenoFile: self.phenoFile = phenoFile = self.fbase+".pheno"
      if not os.path.isfile(phenoFile):
         sys.stderr.write("Could not find phenotype file: %s\n" % (phenoFile))
         return
      f = open(phenoFile,'r')
      keys = []
      P = []
      for line in f:
         v = line.strip().split()
         try:
            P.append(
               [(x.strip() == 'NA' or x.strip() == '-9') and np.nan or float(x)
                for x in v[2:]])
            keys.append((v[0],v[1]))
         except ValueError:
            pass
      if len(P) == 0:
         sys.stderr.write("No valid phenotypes found in file. Exiting")
         sys.exit(1)
      f.close()
      P = np.array(P)
      # reorder to match self.indivs
      D = {}
      L = []
      for i in range(len(keys)): D[keys[i]] = i
      for i in range(len(self.indivs)):
         if not D.has_key(self.indivs[i]): continue
         L.append(D[self.indivs[i]])
      P = P[L,:]
      self.phenos = P
      return P

   def getIndivs(self, base, sampleFile, type='b'):
      if sampleFile is not None: famFile = sampleFile
      elif type == 't': famFile = "%s.tfam" % base
      elif type == 'b': famFile = "%s.fam" % base
      elif type == 'i': famFile = "%s.sample" % base
      else: famFile = ""

      keys = []
      i = 0
      f = open(famFile,'r')
      if( type == 'i' ): f.readline(); f.readline()
      #print famFile
      #print f
      for line in f:
         v = line.strip().split()
         famId = v[0]
         indivId = v[1]
         k = (famId.strip(),indivId.strip())
         keys.append(k)
         i += 1
      f.close()

      self.N = len(keys)
      #sys.stderr.write("Read %d individuals from %s\n" % (self.N, famFile))

      return keys

   def readKinship(self,kFile):
      # Assume the fastLMM style
      # This will read in the kinship matrix and then reorder it
      # according to self.indivs - additionally throwing out individuals
      # that are not in both sets
      if self.indivs == None or len(self.indivs) == 0:
         sys.stderr.write("Did not read any individuals so can't load kinship\n")
         return

      sys.stderr.write("Reading kinship matrix from %s\n" % (kFile) )

      f = open(kFile,'r')
      # read indivs
      if self.fastLMM_kinship_style:
         v = f.readline().strip().split("\t")[1:]
         keys = [tuple(y.split()) for y in v]
         D = {}
         for i in range(len(keys)): D[keys[i]] = i

      # read matrix
      K = []
      if self.fastLMM_kinship_style:
         for line in f:
            K.append([float(x) for x in line.strip().split('\t')[1:]])
      else:
         for line in f:
            K.append([float(x) for x in line.strip().split()])
      f.close()
      K  = np.array(K)


      if self.fastLMM_kinship_style:
         # reorder to match self.indivs
         L = []
         KK = []
         X = []
         for i in range(len(self.indivs)):
            if not D.has_key(self.indivs[i]): X.append(self.indivs[i])
            else:
               KK.append(self.indivs[i])
               L.append(D[self.indivs[i]])
         K = K[L,:][:,L]
         self.indivs = KK
         self.indivs_removed = X
         if len(self.indivs_removed):
            sys.stderr.write("Removed %d individuals that did not appear in"
                             "Kinship\n" % (len(self.indivs_removed)))
      return K

   def getCovariates(self,covFile=None,noMean=False):
      P = None
      if covFile is not None and not os.path.isfile(covFile):
         sys.stderr.write("Could not find covariate file: %s\n" % (covFile))
         sys.exit(1)
      elif covFile is not None:
         f = open(covFile,'r')
         keys = []
         P = []
         for line in f:
            v = line.strip().split()
            keys.append((v[0],v[1]))
            P.append([x == 'NA' and np.nan or float(x) for x in v[2:]])
         f.close()
         P = np.array(P)
         # reorder to match self.indivs
         D = {}
         L = []
         for i in range(len(keys)): D[keys[i]] = i
         for i in range(len(self.indivs)):
            if not D.has_key(self.indivs[i]): continue
            L.append(D[self.indivs[i]])
         P = P[L,:]
      if noMean:
         X0 = P
      elif P is not None:
         X0 = np.hstack([np.ones((self.N,1)),P])
      else:
         X0 = np.ones((self.N,1))
      # P contains just covariates, with no mean vector
      # X0 optionally contains a mean vector
      self.P = P
      self.cov = X0
      return X0
