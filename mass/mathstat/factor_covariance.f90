! This file has three externally callable subroutines:
!
! covprod   compute the product  y = R x  of a covariance matrix with a vector
! covchol   compute the Cholesky factorization  R = U' U  of a covariance matrix
! covsolv   apply the Cholesky factorization to solve linear system R x = y


! Matrix-vector product y = R x, where the covariance matrix R
! R_{ij} = r(|j-i|) = \sum_{m=1}^k a(m) b(m)^|j-i| is real for i,j = 1,2,...
! and |b(m)|<1 for m = 1,...,k.  Memory space for x,y must be disjoint.
!
subroutine covprod(k,a,b,n,x,y)
  implicit none
  integer :: k,n, i,j
  real (8) :: x(n), y(n), s
  complex (8) :: a(k),b(k),ss(k)
  !
  ! Upper triangle
  !
  ss=0
  do i=n, 1, -1
     s=0
     do j=1, k
        ss(j)=ss(j)*b(j)+x(i)
        s=s+a(j)*ss(j)
     end do
     y(i)=s
  end do
  !
  ! (Strict) Lower triangle
  !
  ss=0
  do i=2, n
     s=0
     do j=1, k
        ss(j)=(ss(j)+x(i-1))*b(j)
        s=s+a(j)*ss(j)
     end do
     y(i)=y(i)+s
  end do
end subroutine covprod


! Cholesky decomposition of covariance matrix R, where
! R_{ij} = r(|j-i|) = \sum_{m=1}^k a(m) b(m)^|j-i| is real for i,j = 1,2,...
! and |b(m)|<1 for m = 1,...,k.
!
! The factorization is saved in cholsav, which must have size at least
! 1+(2*k+1)*(n+1) real (8).
!
subroutine covchol(k,a,b,n,cholsav)
  implicit none
  integer :: k,n
  real (8) :: cholsav(*)
  complex (8) :: a(k),b(k)
  cholsav(1)=k
  cholsav(2)=n
  call covcholx(k,a,b,n,cholsav(3),cholsav(3+2*k),cholsav(3+2*k*(1+n)))
end subroutine covchol

subroutine covcholx(k,a,b,n,bsav,d,dsuminv)
  integer :: k,n, i,j,m
  real (8) :: dsuminv(n),s
  complex (8) :: a(k),b(k),bsav(k),d(k,n),bb(k,k),accum(k,k),v(k)
  bsav=b
  s=sum(a)
  d(:,1)=a/sqrt(s)
  dsuminv(1)=1/sum(d(:,1))
  do i=1, k
     do j=1, k
        bb(j,i)=conjg(b(j))*b(i)
        accum(j,i)=conjg(d(j,1))*d(i,1)*bb(j,i)
     end do
  end do
  !
  ! Proceed row by row to factor
  !
  do m=2, n
     do i=1, k
        v(i)=a(i)-sum(accum(:,i))
     end do
     s=sum(v)
     d(:,m)=v/sqrt(s)
     do i=1, k
        do j=1, k
           accum(j,i)=(accum(j,i)+conjg(d(j,m))*d(i,m))*bb(j,i)
        end do
     end do
     dsuminv(m)=1/sum(d(:,m))
  end do
end subroutine covcholx


! Solve linear system R x = y, where R is the covariance matrix described
! above and cholsav was produced by covchol above.  The number of unknowns n
! can be less than or equal to the size for which covchol was created.  The
! solution vector x can coincide in memory with the right hand side y.
!
subroutine covsolv(n,x,y,cholsav)
  implicit none
  integer :: n,k, nsav
  real (8) :: x(n),y(n),cholsav(*)
  k=cholsav(1)
  nsav=cholsav(2)
  call covsolvx(n,x,y,k,cholsav(3),cholsav(3+2*k),cholsav(3+2*k*(nsav+1)))
end subroutine covsolv

subroutine covsolvx(n,x,y,k,b,d,dsuminv)
  implicit none
  integer :: n,k, i
  real (8) :: x(n),y(n),dsuminv(n)
  complex (8) :: b(k),d(k,n),conjb(k),ss(k)
  conjb=conjg(b)
  !
  ! Outer solve
  !
  ss=0
  do i=1, n
     x(i)=(y(i)-sum(ss))*dsuminv(i)
     ss=(ss+x(i)*conjg(d(:,i)))*conjb
  end do
  !
  ! Inner solve
  !
  ss=0
  do i=n, 1, -1
     x(i)=(x(i)-sum(ss*d(:,i)))*dsuminv(i)
     ss=(ss+x(i))*b;
  end do
end subroutine covsolvx


! Matrix-vector product x = U' y, where the covariance matrix R=U'U and
! R_{ij} = r(|j-i|) = \sum_{m=1}^k a(m) b(m)^|j-i| is real for i,j = 1,2,...
! and |b(m)|<1 for m = 1,...,k.  Memory space for x,y must be disjoint.
! CAREFUL!  Joe wrote this one, and he doesn't know F90!
!
subroutine cholprod(n,x,y,cholsav)
  implicit none
  integer :: n,k, nsav
  real (8) :: x(n),y(n),cholsav(*)
  k=cholsav(1)
  nsav=cholsav(2)
  call cholprod2(n,x,y,k,cholsav(3),cholsav(3+2*k))
end subroutine cholprod

subroutine cholprod2(n,x,y,k,b,d)
  implicit none
  integer :: n,k,i,j
  real (8) :: x(n),y(n)
  complex (8) :: b(k),d(k,n),conjb(k),ss(k), s
  conjb=conjg(b)

  ss=0
  do i=1,n
    s=0
    do j=1,k
        ss(j) = ss(j)*conjb(j)+conjg(d(j,i))*y(i)
        s = s+ss(j)
    end do
    x(i) = s
  end do

 end subroutine cholprod2
