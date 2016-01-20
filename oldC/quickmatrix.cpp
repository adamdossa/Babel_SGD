#include "quickmatrix.hpp"

/* Write a character representation of v to fp. */
void VectorPrint(FILE *fp, Vector v)
{
  int i;
  if (fp==stdout) for(i=0;i<v->len;i++) printf("%lg ", v->d[i]);
  else for(i=0;i<v->len;i++) fprintf(fp, "%lg ", v->d[i]);
}

void VectorPrintNL(FILE *fp, Vector v)
{
  VectorPrint(fp, v);
  if (fp==stdout) printf("\n");
  else fprintf(fp, "\n");
}


/* If len < v->len, truncates v.
 * If len > v->len, adds garbage to the end of v. 
 */
void VectorResize(Vector v, int len)
{
  v->d = Reallocate(v->d, len, double);
  v->len = len;
}


/*****************************************************************************/

/* Write a character representation of m to fp. This format was chosen so
 * that row Matrices print just like Vectors.
 */
void MatrixPrint(FILE *fp, Matrix m)
{
  int i,j;
  if (fp==stdout)
    {
      for(i=0;i<m->rows;i++) {
	for(j=0;j<m->cols;j++) {
	  printf("%e ", m->d2[i][j]); 	  // printf("%lg ", m->d2[i][j]);
	}
	printf("\n");
      }
    }
  else
    {
      for(i=0;i<m->rows;i++) {
	for(j=0;j<m->cols;j++) {
	  fprintf(fp,"%lg ", m->d2[i][j]);
	}
	fprintf(fp,"\n");
      }
    }
}


Vector VectorFromData(int length, double *data)
{
  Vector v;

  v = Allocate(1, struct VectorRec);
  v->len = length;
  v->d = data;
  return v;
}

/* Returns the sum of all elements of v. */
double VectorSum(Vector v)
{
  int i;
  double sum = 0.0;
  for(i=0;i<v->len;i++) sum += v->d[i];
  return sum;
}


/* Returns a new, uninitialized Vector whose length is given. */
Vector VectorCreate(int length)
{
  double *data = Allocate(length, double);
  if(!data) {
    printf("Cannot allocate %d element vector\n", length);
    return NULL;
  }
  return VectorFromData(length, data);
}

/* Add vector v2 to vector v1 */
void VectorAdd(Vector v1, Vector v2)
{
  int i;
  for(i=0;i<v1->len;i++) v1->d[i] += v2->d[i];
}

/* Subtract vector v2 from vector v1 */
void VectorSubtract(Vector v1, Vector v2)
{
  int i;
  for(i=0;i<v1->len;i++) v1->d[i] -= v2->d[i];
}


/* Multiply vector v1 by vector v2 */
void VectorMultiply(Vector v1, Vector v2)
{
  int i;
  for(i=0;i<v1->len;i++) v1->d[i] *= v2->d[i];
}


/* Compute vector dot product = VectorSum(VectorMultiply(v1,v2))
 * of v1 and v2. 
 */
double VectorDot(Vector v1, Vector v2)
{
  int i;
  double sum = 0.0;
  for(i=0;i<v1->len;i++) sum += v1->d[i] * v2->d[i];
  return sum;
}

/* Set all elements of v to x. */
void VectorSet(Vector v, double x)
{
  int i;
  for(i=0;i<v->len;i++) v->d[i] = x;
}


/* Multiply all elements of v to x. */
void VectorScale(Vector v, double x)
{
  int i;
  for(i=0;i<v->len;i++) v->d[i] *= x;
}


/* Frees all storage associated with v. */
void VectorFree(Vector v)
{
  free(v->d);
  free(v);
}

/* Returns a Matrix whose data is given. The array becomes property 
 * of the Matrix and will be freed when the Matrix is. The data must be
 * rows*cols elements long.
 */
Matrix MatrixFromData(int rows, int cols, double *data)
{
  Matrix m;
  int i;

  m = Allocate(1, struct MatrixRec);
  m->rows = rows;
  m->cols = cols;
  m->len = rows*cols;
  m->d = data;
  m->d2 = Allocate(rows, double*);
  for(i=0;i<rows;i++) m->d2[i] = data + i*cols;
  return m;
}

/* Returns a new, uninitialized Matrix whose dimensions are given. */
Matrix MatrixCreate(int rows, int cols)
{
  double *data = Allocate(rows*cols, double);
  // #if 0
  if(!data) {
    printf("Cannot allocate %d by %d matrix\n", rows, cols);
    return NULL;
  }
  // #endif
  return MatrixFromData(rows, cols, data);
}

/* Frees all storage associated with m. */
void MatrixFree(Matrix m)
{
  free(m->d2);
  free(m->d);
  free(m);
}

void MatrixVectorMultiply(Vector dest, Matrix a, Vector v)
{
  Vector t = dest;
  int i,j;
  if (dest->len < a->rows) printf("MatrixVectorMultiply Error Need: dest->len >= a->rows\n");
  if (a->cols != v->len) printf("MatrixVectorMultiply Error Need: a->cols == v->len\n");
  if(dest == v) {
    t = VectorCreate(dest->len);
  }
  for(i=0;i<a->rows;i++) {
    double sum = 0;
    for(j=0;j<v->len;j++) {
      sum += a->d2[i][j] * v->d[j];
    }
    t->d[i] = sum;
  }
  if(t != dest) {
    VectorMove(dest, t);
    VectorFree(t);
  }
}


/* Volume implementation.
 * A Volume is a structure of {d, len, xdim, d2, zydim, ydim, d3, zdim}
 */

/* Functions *****************************************************************/

/* Returns a Volume whose data is given. The array becomes property 
 * of the Volume and will be freed when the Volume is. The data must be
 * zdim*ydim*xdim elements long.
 */
Volume VolumeFromData(int zdim, int ydim, int xdim, double *data)
{
  Volume v;
  int i;

  v = Allocate(1, struct VolumeRec);
  v->zdim = zdim;
  v->ydim = ydim;
  v->xdim = xdim;
  v->zydim = zdim*ydim;
  v->len = v->zydim * xdim;

  v->d = data;
  v->d2 = Allocate(v->zydim, double*);
  for(i=0;i<v->zydim;i++) v->d2[i] = &v->d[i*xdim];
  v->d3 = Allocate(v->zdim, double**);
  for(i=0;i<v->zdim;i++) v->d3[i] = &v->d2[i*ydim];
  return v;
}

/* Returns a new, uninitialized Volume whose dimensions are given. */
Volume VolumeCreate(int zdim, int ydim, int xdim)
{
  double *data = Allocate(zdim*ydim*xdim, double);
  if(!data) {
    printf("Cannot allocate %d by %d by %d volume\n", 
	    zdim, ydim, xdim);
    return NULL;
  }
  return VolumeFromData(zdim, ydim, xdim, data);
}

/* Frees all storage associated with v. */
void VolumeFree(Volume v)
{
  free(v->d3);
  free(v->d2);
  free(v->d);
  free(v);
}

/* Returns a new Volume whose data is identical to v. */
Volume VolumeCopy(Volume v)
{
  Volume result = VolumeCreate(v->zdim, v->ydim, v->xdim);
  VolumeMove(result, v);
  return result;
}

/* Returns a new Volume whose data is identical to m
 * and whose shape is zdim by (m->rows/zdim) by m->cols
 */
Volume VolumeFromMatrix(Matrix m, int zdim)
{
  Volume result = VolumeCreate(zdim, m->rows/zdim, m->cols);
  VolumeMove(result, m);
  return result;
}

/* Returns a z slice of a volume. */
Matrix MatrixFromVolume(Volume v, int z)
{
  Matrix m = MatrixCreate(v->ydim, v->xdim);
  memcpy(m->d, v->d3[z][0], m->len*sizeof(double));
  return m;
}

/*****************************************************************************/

/* Returns the grey-scale matrix corresponding to an RGB volume. */
Matrix VolumeGray(Volume v)
{
  Matrix m;
  int i;
  if(v->zdim == 1) return MatrixFromVolume(v, 0);
  m = MatrixCreate(v->ydim, v->xdim);
  for(i=0;i<m->len;i++) {
    m->d[i] = 0.30 * v->d3[0][0][i] + 
              0.59 * v->d3[1][0][i] + 
	      0.11 * v->d3[2][0][i];
  }
  return m;
}

/*****************************************************************************/

/* Write a character representation of v to fp. This format was chosen
 * so that Volumes of depth 1 print just like Matrices.
 */
void VolumePrint(FILE *fp, Volume v)
{
  int i,j,k;
  if (fp==stdout)
    {
      for(i=0;i<v->zdim;i++) {
	for(j=0;j<v->ydim;j++) {
	  for(k=0;k<v->xdim;k++) {
	    printf("%lg ", v->d3[i][j][k]);
	  }
	  printf("\n");
	}
	if(i < v->zdim - 1) printf("\n");
      }
    }
  else
    {
      for(i=0;i<v->zdim;i++) {
	for(j=0;j<v->ydim;j++) {
	  for(k=0;k<v->xdim;k++) {
	    fprintf(fp, "%lg ", v->d3[i][j][k]);
	  }
	  fprintf(fp, "\n");
	}
	if(i < v->zdim - 1) fprintf(fp, "\n");
      }
    }
}

/*****************************************************************************/


Matrix MatrixScan(FILE *fp)
{
  Vector v = VectorCreate(32);
  Matrix mat;
  int rows, cols;
  double d;
  int i = 0;

  /* Use the first line to determine the number of columns */
  for(;;) {
    int c = getc(fp);
    if(c == '\n') break;
    if(c == ' ') continue;   // if(isspace(c)) continue;
    ungetc(c, fp);
    if(fscanf(fp, "%lg", &d) < 1) {
      printf("MatrixScan expected a real number\n");
      VectorFree(v);
      return NULL;
    }
    if(i == v->len) {
      VectorResize(v, v->len * 2);
    }
    v->d[i++] = d;
  }
  cols = i;
  
  /* Read the rest of the lines */
  for(rows=1;;rows++) {
    int j;
    for(j = 0; j < cols; j++) {
      if(fscanf(fp, "%lg", &d) < 1) {
	if(!feof(fp)) {
	  printf("MatrixScan expected a real number\n");
	  VectorFree(v);
	  return NULL;
	}
	else if(j > 0) {
	  printf("unexpected end of file in MatrixScan");
	  VectorFree(v);
	  return NULL;
	}
	break;
      }
      if(i == v->len) {
	VectorResize(v, v->len * 2);
      }
      v->d[i++] = d;
    }
    if(feof(fp)) break;
  }
  if(rows*cols != v->len) {
    VectorResize(v, rows*cols);
  }
  mat = MatrixFromData(rows, cols, v->d);
  free(v);
  return mat;
}

Matrix MatrixReadBinary(FILE *fp)
{
  Matrix m;
  int rows, cols;
  /* read rows and cols */
  if(fscanf(fp, "%d %d", &rows, &cols) < 2) {
    printf("bad matrix header\n");
    return NULL;
  }
  /* eat a newline */
  while(fgetc(fp) != '\n');
  m = MatrixCreate(rows, cols);
  if(((int) fread(m->d, sizeof(double), m->len, fp)) < m->len) {
    printf("eof while reading matrix data (%d by %d)\n", rows, cols);
    MatrixFree(m);
    return NULL;
  }
  /* AdjustBuffer(m->d, m->len, sizeof(double)); */
  return m;
}


/* Returns a new Matrix whose data is identical to m. */
Matrix MatrixCopy(Matrix m)
{
  Matrix result = MatrixCreate(m->rows, m->cols);
  MatrixMove(result, m);
  return result;
}


void MatrixReshape(Matrix m, int rows, int cols)
{
  int i;
  m->rows = rows;
  m->cols = cols;
  Reallocate(m->d2, m->rows, double*);
  for(i=0;i<m->rows;i++) m->d2[i] = m->d + i*m->cols;
}


/* Transposes a matrix in place. */
void MatrixTranspose(Matrix m)
{
  int i,j;
  if(m->rows == m->cols) {
    for(i=0;i<m->rows;i++) {
      for(j=0;j<i;j++) {
	double t = m->d2[i][j];
	m->d2[i][j] = m->d2[j][i];
	m->d2[j][i] = t;
      }
    }
  }
  else {
    Matrix t = MatrixCopy(m);
    MatrixReshape(m, t->cols, t->rows);
    for(i=0;i<m->rows;i++) {
      for(j=0;j<m->cols;j++) {
	m->d2[i][j] = t->d2[j][i];
      }
    }
    MatrixFree(t);
  }
}

/* Modifies dest to be the matrix product ab. 
 * dest may be the same as a or b (or both).
 * Requires: dest->rows == a->rows, dest->cols == b->cols, a->cols == b->rows
 */
void MatrixMultiply(Matrix dest, Matrix a, Matrix b)
{
  double sum;
    int i,j,k;
  if (dest->rows != a->rows) printf("Matrix Multiply Error, need dest->rows == a->rows \n");
  if (dest->cols != b->cols) printf("Matrix Multiply Error, need dest->cols == b->cols \n");
  if (a->cols != b->rows) printf("Matrix Multiply Error, need a->cols == b->rows \n");
  if((dest == a) && (dest != b)) {
    /* dest = dest*b */
    Vector row = VectorCreate(dest->cols);
    for(i=0;i<dest->rows;i++) {
      for(j=0;j<dest->cols;j++) {
	double sum = 0;
	for(k=0;k<b->rows;k++) {
	  sum += a->d2[i][k] * b->d2[k][j];
	}
	row->d[j] = sum;
      }
      /* clobber row i of a only after we are done with it */
      memcpy(dest->d2[i], row->d, row->len*sizeof(double));
    }
    VectorFree(row);
  }
  else if((dest == b) && (dest != a)) {
    /* dest = a*dest */
    Vector col = VectorCreate(dest->rows);
    for(i=0;i<dest->cols;i++) {
      for(j=0;j<dest->rows;j++) {
	double sum = 0;
	for(k=0;k<a->cols;k++) {
	  sum += a->d2[j][k] * b->d2[k][i];
	}
	col->d[j] = sum;
      }
      /* clobber col i of b only after we are done with it */
      for(j=0;j<dest->rows;j++) {
	dest->d2[j][i] = col->d[j];
      }
    }
    VectorFree(col);
  }
  else if((dest == a) && (dest == b)) {
    /* dest = a^2 */
    Matrix t = MatrixCopy(a);
    MatrixMultiply(dest, a, t);
    MatrixFree(t);
  }
  else {    
    for(i=0;i<dest->rows;i++) {
        for(j=0;j<dest->cols;j++) {
            sum = 0;
            for(k=0;k<b->rows;k++) {
              sum += a->d2[i][k] * b->d2[k][j];
            }
            dest->d2[i][j] = sum;
        }
    }
  }
}

/* Modifies dest to be the matrix product ab'. 
 * dest may be the same as a or b' (or both).
 * Requires: dest->rows == a->rows, dest->cols == b->rows, a->cols == b->cols
 */
void MatrixMultiplyTranspose(Matrix dest, Matrix a, Matrix b)
{
  int i,j,k;
  //  assert(dest->rows == a->rows);
  //  assert(dest->cols == b->cols);
  //  assert(a->cols == b->rows);
  if((dest == a) && (dest != b)) {
    /* dest = dest*b' */
    Vector row = VectorCreate(dest->cols);
    for(i=0;i<dest->rows;i++) {
      for(j=0;j<dest->cols;j++) {
	double sum = 0;
	for(k=0;k<b->cols;k++) {
	  sum += a->d2[i][k] * b->d2[j][k];
	}
	row->d[j] = sum;
      }
      /* clobber row i of a only after we are done with it */
      memcpy(dest->d2[i], row->d, row->len*sizeof(double));
    }
    VectorFree(row);
  }
  else if((dest == b) && (dest != a)) {
    /* dest = a*dest' */
    Vector col = VectorCreate(dest->rows);
    for(i=0;i<dest->rows;i++) {
      for(j=0;j<dest->cols;j++) {
	double sum = 0;
	for(k=0;k<a->cols;k++) {
	  sum += a->d2[j][k] * b->d2[i][k];
	}
	col->d[j] = sum;
      }
      /* clobber col i of b only after we are done with it */
      for(j=0;j<dest->cols;j++) {
	dest->d2[i][j] = col->d[j];
      }
    }
    VectorFree(col);
  }
  else if((dest == a) && (dest == b)) {
    /* dest = a*a' */
    Matrix t = MatrixCopy(a);
    MatrixTranspose(t);
    MatrixMultiply(dest, a, t);
    MatrixFree(t);
  }
  else {
    for(i=0;i<dest->rows;i++) {
      for(j=0;j<dest->cols;j++) {
	double sum = 0;
	for(k=0;k<b->cols;k++) {
	  sum += a->d2[i][k] * b->d2[j][k];
	}
	dest->d2[i][j] = sum;
      }
    }
  }
}

static double at,bt,ct;
#define PYTHAG(a,b) ((at=fabs(a)) > (bt=fabs(b)) ? \
(ct=bt/at,at*sqrt(1.0+ct*ct)) : (bt ? (ct=at/bt,bt*sqrt(1.0+ct*ct)): 0.0))

static double maxarg1,maxarg2;
#define MAXT(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ?\
	(maxarg1) : (maxarg2))
#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

int compar(const void* a_, const void* b_)
{
  double a=*(double*)a_;
  double b=*(double*)b_;
  return((a<b)?1:((a>b)?-1:0));
}


//
// Singular value decomposition
//
int MatrixSVD(Matrix A, Matrix U, Matrix V, Vector S)
{
  //
  // Map arguments
  // A is copied to U preserve original contents
  //
  int      m=A->rows;
  int      n=A->cols;
  if (!U || U->rows!=m || U->cols!=n) {
    printf("U not allocated or not size of A\n");
  }
  MatrixMove(U,A);
  double** a=U->d2;
  double *w=S->d;
  double **v=V->d2;

  int flag,i,its,j,jj,k,l,nm;
  double c,f,h,s,x,y,z;
  double anorm=0.0,g=0.0,scale=0.0;
  double *rv1;

  if (m < n) printf("SVDCMP: You must augment A with extra zero rows\n");
  rv1=new double[n];
  for (i=0;i<n;i++) {
    l=i+1;
    rv1[i]=scale*g;
    g=s=scale=0.0;
    if (i < m) {
      for (k=i;k<m;k++) scale += fabs(a[k][i]);
      if (scale) {
	for (k=i;k<m;k++) {
	  a[k][i] /= scale;
	  s += a[k][i]*a[k][i];
	}
	f=a[i][i];
	g = -SIGN(sqrt(s),f);
	h=f*g-s;
	a[i][i]=f-g;
	if (i != n) {
	  for (j=l;j<n;j++) {
	    for (s=0.0,k=i;k<m;k++) s += a[k][i]*a[k][j];
	    f=s/h;
	    for (k=i;k<m;k++) a[k][j] += f*a[k][i];
	  }
	}
	for (k=i;k<m;k++) a[k][i] *= scale;
      }
    }
    w[i]=scale*g;
    g=s=scale=0.0;
    if (i < m && i != n-1) {
      for (k=l;k<n;k++) scale += fabs(a[i][k]);
      if (scale) {
	for (k=l;k<n;k++) {
	  a[i][k] /= scale;
	  s += a[i][k]*a[i][k];
	}
	f=a[i][l];
	g = -SIGN(sqrt(s),f);
	h=f*g-s;
	a[i][l]=f-g;
	for (k=l;k<n;k++) rv1[k]=a[i][k]/h;
	if (i != m-1) {
	  for (j=l;j<m;j++) {
	    for (s=0.0,k=l;k<n;k++) s += a[j][k]*a[i][k];
	    for (k=l;k<n;k++) a[j][k] += s*rv1[k];
	  }
	}
	for (k=l;k<n;k++) a[i][k] *= scale;
      }
    }
    anorm=MAXT(anorm,(fabs(w[i])+fabs(rv1[i])));
  }
  for (i=n-1;i>=0;i--) {
    if (i < n-1) {
      if (g) {
	for (j=l;j<n;j++)
	  v[j][i]=(a[i][j]/a[i][l])/g;
	for (j=l;j<n;j++) {
	  for (s=0.0,k=l;k<n;k++) s += a[i][k]*v[k][j];
	  for (k=l;k<n;k++) v[k][j] += s*v[k][i];
	}
      }
      for (j=l;j<n;j++) v[i][j]=v[j][i]=0.0;
    }
    v[i][i]=1.0;
    g=rv1[i];
    l=i;
  }
  for (i=n-1;i>=0;i--) {
    l=i+1;
    g=w[i];
    if (i < n-1)
      for (j=l;j<n;j++) a[i][j]=0.0;
    if (g) {
      g=1.0/g;
      if (i != n-1) {
	for (j=l;j<n;j++) {
	  for (s=0.0,k=l;k<m;k++) s += a[k][i]*a[k][j];
	  f=(s/a[i][i])*g;
	  for (k=i;k<m;k++) a[k][j] += f*a[k][i];
	}
      }
      for (j=i;j<m;j++) a[j][i] *= g;
    } else {
      for (j=i;j<m;j++) a[j][i]=0.0;
    }
    ++a[i][i];
  }
  for (k=n-1;k>=0;k--) {
    for (its=0;its<30;its++) {
      flag=1;
      for (l=k;l>=0;l--) {
	nm=l-1;
	if (fabs(rv1[l])+anorm == anorm) {
	  flag=0;
	  break;
	}
	if (fabs(w[nm])+anorm == anorm) break;
      }
      if (flag) {
	c=0.0;
	s=1.0;
	for (i=l;i<k;i++) {
	  f=s*rv1[i];
	  if (fabs(f)+anorm != anorm) {
	    g=w[i];
	    h=PYTHAG(f,g);
	    w[i]=h;
	    h=1.0/h;
	    c=g*h;
	    s=(-f*h);
	    for (j=0;j<m;j++) {
	      y=a[j][nm];
	      z=a[j][i];
	      a[j][nm]=y*c+z*s;
	      a[j][i]=z*c-y*s;
	    }
	  }
	}
      }
      z=w[k];
      if (l == k) {
	if (z < 0.0) {
	  w[k] = -z;
	  for (j=0;j<n;j++) v[j][k]=(-v[j][k]);
	}
	break;
      }
      if (its == 29) printf("No convergence in 30 SVDCMP iterations\n");
      x=w[l];
      nm=k-1;
      y=w[nm];
      g=rv1[nm];
      h=rv1[k];
      f=((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
      g=PYTHAG(f,1.0);
      f=((x-z)*(x+z)+h*((y/(f+SIGN(g,f)))-h))/x;
      c=s=1.0;
      for (j=l;j<=nm;j++) { // think it's OK
	i=j+1;
	g=rv1[i];
	y=w[i];
	h=s*g;
	g=c*g;
	z=PYTHAG(f,h);
	rv1[j]=z;
	c=f/z;
	s=h/z;
	f=x*c+g*s;
	g=g*c-x*s;
	h=y*s;
	y=y*c;
	for (jj=0;jj<n;jj++) {
	  x=v[jj][j];
	  z=v[jj][i];
	  v[jj][j]=x*c+z*s;
	  v[jj][i]=z*c-x*s;
	}
	z=PYTHAG(f,h);
	w[j]=z;
	if (z) {
	  z=1.0/z;
	  c=f*z;
	  s=h*z;
	}
	f=(c*g)+(s*y);
	x=(c*y)-(s*g);
	for (jj=0;jj<m;jj++) {
	  y=a[jj][j];
	  z=a[jj][i];
	  a[jj][j]=y*c+z*s;
	  a[jj][i]=z*c-y*s;
	}
      }
      rv1[l]=0.0;
      rv1[k]=f;
      w[k]=x;
    }
  }
  delete(rv1);
  //
  // Sort singular values descending
  //
  double *index;
  index = new double[2*n];

  for (unsigned i=0; i<n; i++) {
    index[2*i]=S->d[i];
    index[2*i+1]=i;
  }
  qsort(index,n,2*sizeof(double),compar);
  //
  // Recover the permutation
  //
  unsigned *perm;
  perm = new unsigned[n];
  for (unsigned i=0; i<n; i++) {
    perm[i]=(int)index[2*i+1];
  }
  delete(index);
  //
  // Permute U,V,S to match sorted order
  // Transpose U,V for simplicity
  //
  MatrixTranspose(U);
  MatrixTranspose(V);
  double tmp_S;
  double *tmp_U;
  tmp_U = new double[U->cols];
  double *tmp_V;
  tmp_V = new double[V->cols];
  tmp_S=S->d[0];
  memmove(tmp_U,U->d2[0],U->cols*sizeof(double));
  memmove(tmp_V,V->d2[0],V->cols*sizeof(double));
  unsigned ix=0;
  do {
    //
    // Move data between elements
    //
    S->d[ix]=S->d[perm[ix]];
    memmove(U->d2[ix],U->d2[perm[ix]],U->cols*sizeof(double));
    memmove(V->d2[ix],V->d2[perm[ix]],V->cols*sizeof(double));
    //
    // Move to next element in permutation
    //
    ix=perm[ix];
    //
    // When source is element 0, fall out
    //
  } while (perm[ix]);
  //
  // Move final element
  //
  delete(perm);

  S->d[ix]=tmp_S;
  memmove(U->d2[ix],tmp_U,U->cols*sizeof(double));
  memmove(V->d2[ix],tmp_V,V->cols*sizeof(double));
  delete(tmp_U);
  delete(tmp_V);
  //
  // Transpose U,V back
  //
  MatrixTranspose(U);
  MatrixTranspose(V);
  return 0;
}


double MatrixDeterminant(Matrix A)
{

  //
  // Implement determinant using SVD
  //
  Matrix U=MatrixCreate(A->rows,A->cols);
  Matrix V=MatrixCreate(A->cols,A->cols);
  Vector s=VectorCreate(A->cols);
  MatrixSVD(A,U,V,s);

  int i;
  double determ=1.0;
  for(i=0;i<s->len;i++) determ *= s->d[i];

  MatrixFree(U);
  MatrixFree(V);
  VectorFree(s);
  return(determ);
}

int MatrixPseudoInvert(Matrix A)
{
  //
  // Implement pseudo invert using SVD
  //
  Matrix U=MatrixCreate(A->rows,A->cols);
  Matrix V=MatrixCreate(A->cols,A->cols);
  Matrix S=MatrixCreate(A->cols,A->cols);
  Vector s=VectorCreate(A->cols);
  
  //
  // Call SVD
  //
  MatrixSVD(A,U,V,s);
  //
  // Put reciprocal of s on diagonal of S
  //
  for (unsigned i=0; i<S->rows; i++) {
    for (unsigned j=0; j<S->cols; j++) {
      S->d2[i][j]=(i==j&&s->d[i]>1e-6)?1.0/s->d[i]:0.0;
    }
  }
  //
  // Reshape A for inverse
  //
  if (A->cols!=A->rows) {
    MatrixReshape(A,A->cols,A->rows);
  }
  //
  // Transpose U
  //
  MatrixTranspose(U);
  //
  // A^{-1}=VS^{-1}U^T
  //
  MatrixMultiply(S,V,S);
  MatrixMultiply(A,S,U);
  //
  // Free matrices and vector
  //
  MatrixFree(U);
  MatrixFree(V);
  MatrixFree(S);
  VectorFree(s);
  return 0;
}


int MatrixInvert(Matrix A)
{
  //
  // Test for squareness
  //
  if (A->rows != A->cols) {
    printf("MatrixInvert: matrix must be square\n");
    return(-1);
  }
  //
  // Implement invert as pseudo invert
  //
  return(MatrixPseudoInvert(A));
}


int MatrixEigenvectorsSymTony(Matrix A, double *v)
{
  // Returns oldA = A'*diag(v)*A;
  //

  // Test allocation
  //
  if (!A || !v) {
    printf("MatrixEigenvectorsSymTony: NULL matrix or array\n");
    return(-1);
  }
  //
  // Test for squareness
  //
  if (A->rows != A->cols) {
    printf("MatrixEigenvectorsSymTony: matrix must be square\n");
    return(-1);
  }
  //
  // Test for symmetry
  //
  for (unsigned i=0; i< A->rows; i++) {
    for (unsigned j=i+1; j< A->cols; j++) {
      if (A->d2[i][j] != A->d2[j][i]) {
	printf("MatrixEigenvectorsSymTony: matrix must be symmetric, as below:\n");
	MatrixPrint(stdout,A);
	printf("\n Verify the above \n");
	return(-1);  
      }
    }
  }
  //
  // Implement EigenvectorsSym using SVD
  //
  Matrix U=MatrixCreate(A->rows,A->cols);
  Matrix V=MatrixCreate(A->cols,A->cols);
  Vector s=VectorCreate(A->cols);
  //
  // Call SVD
  //
  MatrixSVD(A,U,V,s);
  //
  // Test for negative Eigenvalues
  //
  // double eigdot[A->cols]; // TONY replace eigdot with v
  for (unsigned j=0; j<A->cols; j++) {
    v[j]=0.0;
    for (unsigned i=0; i<A->rows; i++) {
      v[j]+=U->d2[i][j]*V->d2[i][j];
    }
  }
  MatrixMove(V,A); // Make V the original A matrix
  //
  // Flip sign if necessary
  // Transpose Eigenvectors into A as caller expects
  // Copy Eigenvalues from s
  //
  for (unsigned j=0; j<A->cols; j++) {
    double tmpv = v[j];
    v[j]=(tmpv<0)?-s->d[j]:s->d[j];
    for (unsigned i=0; i<A->rows; i++) {
      A->d2[j][i]=(tmpv<0)?-U->d2[i][j]:U->d2[i][j];
    }
  }
  MatrixMultiply(U,A,V);
  MatrixTranspose(A);
  MatrixMultiply(U,U,A);
  MatrixTranspose(A);
  for (unsigned j=0; j<A->cols; j++) v[j]=U->d2[j][j];

  //
  // Free matrices and vector
  //
  MatrixFree(U);
  MatrixFree(V);
  VectorFree(s);
  return 0;
}

//
// Returns Eigenvectors in A and Eigenvalues in v
// Assumes A is symmetric so Eigenvalues are all real.
//
int MatrixEigenvectorsSym(Matrix A, double *v)
{
  // Test allocation
  //
  if (!A || !v) {
    printf("MatrixEigenvectorsSym: NULL matrix or array\n");
    return(-1);
  }
  //
  // Test for squareness
  //
  if (A->rows != A->cols) {
    printf("MatrixEigenvectorsSym: matrix must be square\n");
    return(-1);
  }
  //
  // Test for symmetry
  //
  for (unsigned i=0; i< A->rows; i++) {
    for (unsigned j=i+1; j< A->cols; j++) {
      if (A->d2[i][j] != A->d2[j][i]) {
	printf("MatrixEigenvectorsSym: matrix must be symmetric, as below:\n");
	MatrixPrint(stdout,A);
	printf("\n Verify the above \n");
	return(-1);  
      }
    }
  }
  //
  // Implement EigenvectorsSym using SVD
  //
  Matrix U=MatrixCreate(A->rows,A->cols);
  Matrix V=MatrixCreate(A->cols,A->cols);
  Vector s=VectorCreate(A->cols);
  //
  // Call SVD
  //
  MatrixSVD(A,U,V,s);
  //
  // Test for negative Eigenvalues
  //
  // double eigdot[A->cols]; // TONY replace eigdot with v
  for (unsigned j=0; j<A->cols; j++) {
    v[j]=0.0;
    for (unsigned i=0; i<A->rows; i++) {
      v[j]+=U->d2[i][j]*V->d2[i][j];
    }
  }
  //
  // Flip sign if necessary
  // Transpose Eigenvectors into A as caller expects
  // Copy Eigenvalues from s
  //
  for (unsigned j=0; j<A->cols; j++) {
    double tmpv = v[j];
    v[j]=(tmpv<0)?-s->d[j]:s->d[j];
    for (unsigned i=0; i<A->rows; i++) {
      A->d2[j][i]=(tmpv<0)?-U->d2[i][j]:U->d2[i][j];
    }
  }
  //
  // Free matrices and vector
  //
  MatrixFree(U);
  MatrixFree(V);
  VectorFree(s);
  return 0;
}


int MatrixEigenvectorsSymX(Matrix A, double *v, int count, Matrix V)
{
  //
  // Make a duplicate of A
  //
  Matrix VV=MatrixCreate(A->rows,A->cols);
  MatrixMove(VV,A);
  //
  // Allocate space for eigenvalues
  //
  double *vv;
  vv = new double[A->cols];
  //
  // Implement EigenvectorsSymX using MatrixEigenvectorsSym
  //
  MatrixEigenvectorsSym(VV,vv);
  //
  // Copy top count values and vectors
  //
  for (unsigned i=0; i<count; i++) {
    v[i]=vv[i];
    for (unsigned j=0; j<V->cols; j++) {
      V->d2[i][j]=VV->d2[i][j];
    }
  }
  //
  // Free memory
  //
  delete(vv);
  MatrixFree(VV);

  return 0;
}


//
// Returns Eigenvectors in A and real Eigenvalues in vr
// Assumes A is symmetric so Eigenvalues are all real.
//
int MatrixEigenvalues(Matrix A, double *vr, double*)
{
  //
  // Implement Eigenvalues using MatrixEigenvectorsSym
  //
  MatrixEigenvectorsSym(A,vr);
  return 0;
}
