def work_odd_or_even():
    '''
    1에서 1000 사이의 숫자를 입력받아 홀수인지 짝수인지 확인한 후 메시지를 출력함
    문자열 'No'를 입력하면 종료함
    '''

    strnum = input("Enter an integer between 1 and 1000. (To quit, enter 'No'.) >>> ")

    while True:
        if strnum == 'No':
            print("Have a nice day!")
            return
        else:
            try:
                num = int(strnum)
            except:
                print("You should enter an integer or say 'No'.\n")
                strnum = input("Have another? (If not, enter 'No'.) >>> ")
                continue

            if not 1 <= num <= 1000:
                print("You should enter an integer between 1 and 1000.\n")
            elif num % 2:
                print("That's an odd number!\n")
            else:
                print("That's an even number!\n")
            
            strnum = input("Have another? (If not, enter 'No'.) >>> ")

work_odd_or_even()