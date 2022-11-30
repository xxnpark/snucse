#----------------------------------------------------------------
#
#  4190.308 Computer Architecture (Fall 2022)
#
#  Project #3: Image Resizing in RISC-V Assembly
#
#  November 20, 2022
# 
#  Seongyeop Jeong (seongyeop.jeong@snu.ac.kr)
#  Jaehoon Shim (mattjs@snu.ac.kr)
#  IlKueon Kang (kangilkueon@snu.ac.kr)
#  Wookje Han (gksdnrwp@snu.ac.kr)
#  Jinsol Park (jinsolpark@snu.ac.kr)
#  Systems Software & Architecture Laboratory
#  Dept. of Computer Science and Engineering
#  Seoul National University
#
#----------------------------------------------------------------

####################
# void bmpresize(unsigned char *imgptr, int h, int w, int k, unsigned char *outptr)
# a0 imgptr / a1 h / a2 w / a3 k / a4 outptr
# use x0, sp, ra, a0 ~ a4, t0 ~ t4
####################

	.globl bmpresize
bmpresize:
	addi	sp, sp, -44
	sw		a0, 0(sp)		# sp[0] = imgptr
#	sw		a1, 4(sp)		# sp[1] = h
#	sw		a2, 8(sp)		# sp[2] = w
	sw		a3, 12(sp)		# sp[3] = k
	sw		a4, 16(sp)		# sp[4] = outptr
#	sw		ra, 20(sp)		# sp[5] = ra

	srl		t3, a1, a3		# t3 = h >> k : hr
	sw		t3, 24(sp)		# sp[6] = hr
	srl		t0, a2, a3		# t0 = w >> k : wr
	sll		t1, t0, 1		# t1 = wr * 2
	add		t0, t0, t1		# t0 = wr * 3
	sw		t0, 28(sp)		# sp[7] = wr * 3

	slli 	t0, a2, 1		# t0 = w * 2
	add 	t0, t0, a2		# t0 = w * 3
	addi	t1, t0, 3		# t1 = w * 3 + 3
	srli	t1, t1, 2		# t1 = (w * 3 + 3) / 4 : wl
	sw		t1, 32(sp)		# sp[8] = wl
	srl		t2, t0, a3		# t2 = (w * 3) >> k
	addi	t2, t2, 3		# t2 = (w * 3) >> k + 3
	srli	t4, t2, 2		# t4 = ((w * 3) >> k + 3) / 4 : wrl
	sw		t4, 36(sp)		# sp[9] = wrl

	addi	t0, x0, 1		# t0 = 1
	sll		t0, t0, a3		# t0 = 2^k
	sw		t0, 40(sp)		# sp[10] = 2^k

	add		t0, x0, x0		# t0 = 0
	add		t1, x0, x0		# iterator for mhrwrl
mhrwrl:						# multiply hr and wrl
	add		t0, t0, t4		# t0 = t0 + wrl
	add		t1, t1, 1
	blt		t1, t3, mhrwrl
	# end mhrwrl			# t0 = hr * wrl

	lw		t4, 16(sp)		# t4 = outptr
	add		t1, x0, x0		# iterator for ioutptr
ioutptr:					# initialize outptr
	slli	t2, t1, 2		# t2 = t1 * 4
	add		t3, t4, t2		# t3 = outptr + t1 * 4
	sw		x0, 0(t3)		# outptr[t1 * 4] = 0x00000000
	add		t1, t1, 1
	blt		t1, t0, ioutptr
	# end ioutptr

	add		t1, x0, x0		# t1 = 0 : i							# t1=i
L1:
	add		t2, x0, x0		# t2 = 0 : j							# t1=i, t2=j	
L2:
	add		a0, x0, x0		# a0 = 0 : sum							# t1=i, t2=j, a0=sum
	add		t3, x0, x0		# t3 = 0 : hi							# t1=i, t2=j, t3=hi, a0=sum
L3:
	lw		a1, 12(sp)		# a1 = k
	sll		a1, t1, a1		# a1 = i * 2^k
	add		a1, a1, t3		# a1 = i * 2^k + hi : ii				# t1=i, t2=j, t3=hi, a0=sum, a1==ii
	add		t4, x0, x0		# t4 = 0 : hj							# t1=i, t2=j, t3=hi, t4=hj, a0=sum, a1=ii
L4:
	addi	a2, t2, 0		# a2 = j
	add		t0, x0, -1		# iterator for djt
djt:						# divide j by three
	addi	a2, a2, -3		# a2 = a2 - 3
	addi	t0, t0, 1
	bge		a2, x0, djt
	addi	a2, a2, 3
	# end djt				# t0 = j / 3, a2 = j % 3				# t0=j/3, t1=i, t2=j, t3=hi, t4=hj, a0=sum, a1=ii, a2=j%3

	slli	a3, t0, 1		# a3 = (j / 3) * 2
	add		t0, t0, a3		# t0 = (j / 3) * 3
	lw		a3, 12(sp)		# a3 = k
	sll		t0, t0, a3		# to = ((j / 3) * 3) * 2^k
	add		a2, a2, t0		# a2 = ((j / 3) * 3) * 2^k + j % 3		# t1=i, t2=j, t3=hi, t4=hj, a0=sum, a1=ii, a2=((j/3)*3)*2^k+j%3
	slli	t0, t4, 1		# t0 = hj * 2
	add		t0, t0, t4		# t0 = hj * 3
	add		a2, a2, t0		# a2 = ((j / 3) * 3) * 2^k + j % 3 + hj * 3 : jj	# t1=i, t2=j, t3=hi, t4=hj, a0=sum, a1=ii, a2=jj

	srli	a3, a2, 2		# a3 = jj / 4
	add		t0, x0, x0		# iterator for giwi
giwi:
	add		a3, a3, a1		# a3 = a3 + ii
	addi	t0, t0, 1
	lw		a4, 32(sp)		# a4 = wl
	blt		t0, a4, giwi
	# end giwi				# a3 = jj / 4 + ii * wl : ind			# t1=i, t2=j, t3=hi, t4=hj, a0=sum, a1=ii, a2=jj, a3=inind
	lw		t0, 0(sp)		# t0 = imgptr
	slli	a3, a3, 2		# a3 = inind * 4
	add		t0, t0, a3		# t0 = imgptr + inind * 4
	lw		a3, 0(t0)		# a3 = imgptr[inind] : inword			# t1=i, t2=j, t3=hi, t4=hj, a0=sum, a1=ii, a2=jj, a3=inword

	srli	a4, a2, 2		# a4 = jj / 4
	slli	t0, a4, 2		# t0 = (jj / 4) * 4
	sub		a4, a2, t0		# a4 = jj % 4 : jjt						# t1=i, t2=j, t3=hi, t4=hj, a0=sum, a1=ii, a2=jj, a3=jj/4, a4=jjt
	slli	a4, a4, 3		# a4 = jjt * 8
	srl		a3, a3, a4		# a3 = inword / 256^jjt
	srli	t0, a3, 8		# t0 = (inword / 256^jjt) / 256
	slli	t0, t0, 8		# t0 = ((inword / 256^jjt) / 256) * 256
	sub		a3, a3, t0		# a3 = (inword / 256^jjt) % 256 : inval	# t1=i, t2=j, t3=hi, t4=hj, a0=sum, a1=ii, a2=jj, a3=inval
	add		a0, a0, a3		# a0 = a0 + inval						# t1=i, t2=j, t3=hi, t4=hj, a0=sum, a1=ii, a2=jj

	addi	t4, t4, 1
	lw		t0, 40(sp)		# t0 = 2^k
	blt		t4, t0, L4
	# end L4
	addi	t3, t3, 1
	blt		t3, t0, L3
	# end L3														# t1=i, t2=j, a0=sum, a1=ii, a2=jj

	lw		t0, 12(sp)		# t0 = k
	slli	t0, t0, 1		# t0 = k * 2
	srl		a0, a0, t0		# a0 = sum >> (k * 2) : avg				# t1=i, t2=j, a0=avg, a1=ii, a2=jj

	srli	a3, t2, 2		# a3 = j / 4
	slli	a4, a3, 2		# a4 = (j / 4) * 4
	sub		a4, t2, a4		# a4 = j % 4 : jt						# t1=i, t2=j, a0=avg, a1=ii, a2=jj, a3=j/4, a4=jt
	add		t0, x0, x0		# iterator for gowi
gowi:
	add		a3, a3, t1		# a3 = a3 + i
	addi	t0, t0, 1
	lw		t3, 36(sp)		# a4 = wrl
	blt		t0, t3, gowi
	# end gowi				# a3 = j / 4 + i * wrl					# t1=i, t2=j, a0=avg, a1=ii, a2=jj, a3=outind, a4=jt

	lw		t0, 16(sp)		# t0 = outptr
	slli	a3, a3, 2		# a3 = outind * 4
	add		t3, t0, a3		# t3 = outptr + outind * 4 : outiter
 	lw		a3, 0(t3)		# a3 = outptr[outind] : outword			# t1=i, t2=j, t3=outiter, a0=avg, a1=ii, a2=jj, a3=outword, a4=jt

	slli	a4, a4, 3		# a4 = jt * 8
	sll		t0, a0, a4		# t0 = avg * 256^jt : newoutval
 	add		a3, a3, t0		# a3 = outword + newoutval : newoutword
 	sw		a3, 0(t3)		# outptr[outind] = newoutword

	addi	t2, t2, 1
	lw		a3, 28(sp)		# a2 = wr * 3
	blt		t2, a3, L2		# 3wr번 반복
	# end L2

	add 	t1, t1, 1
	lw		a3, 24(sp)		# a3 = hr
	blt		t1, a3, L1		# hr번 반복
	# end L1
	
	addi	sp, sp, 44

	ret
