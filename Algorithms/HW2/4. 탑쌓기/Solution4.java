import java.util.StringTokenizer;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.PrintWriter;

/*
   1. 아래와 같은 명령어를 입력하면 컴파일이 이루어져야 하며, Solution4 라는 이름의 클래스가 생성되어야 채점이 이루어집니다.
       javac Solution4.java -encoding UTF8


   2. 컴파일 후 아래와 같은 명령어를 입력했을 때 여러분의 프로그램이 정상적으로 출력파일 output4.txt 를 생성시켜야 채점이 이루어집니다.
       java Solution4

   - 제출하시는 소스코드의 인코딩이 UTF8 이어야 함에 유의 바랍니다.
   - 수행시간 측정을 위해 다음과 같이 time 명령어를 사용할 수 있습니다.
       time java Solution4
   - 일정 시간 초과시 프로그램을 강제 종료 시키기 위해 다음과 같이 timeout 명령어를 사용할 수 있습니다.
       timeout 0.5 java Solution4   // 0.5초 수행
       timeout 1 java Solution4     // 1초 수행
 */

class Solution4 {
	static final int max_n = 1000;

	static int n, H;
	static int[] h = new int[max_n], d = new int[max_n-1];
	static int Answer;

	public static void main(String[] args) throws Exception {
		/*
		   동일 폴더 내의 input4.txt 로부터 데이터를 읽어옵니다.
		   또한 동일 폴더 내의 output4.txt 로 정답을 출력합니다.
		 */
		BufferedReader br = new BufferedReader(new FileReader("input4.txt"));
		StringTokenizer stk;
		PrintWriter pw = new PrintWriter("output4.txt");

		/*
		   10개의 테스트 케이스가 주어지므로, 각각을 처리합니다.
		 */
		for (int test_case = 1; test_case <= 10; test_case++) {
			/*
			   각 테스트 케이스를 표준 입력에서 읽어옵니다.
			   먼저 블록의 개수와 최대 높이를 각각 n, H에 읽어들입니다.
			   그리고 각 블록의 높이를 h[0], h[1], ... , h[n-1]에 읽어들입니다.
			   다음 각 블록에 파인 구멍의 깊이를 d[0], d[1], ... , d[n-2]에 읽어들입니다.
			 */
			stk = new StringTokenizer(br.readLine());
			n = Integer.parseInt(stk.nextToken()); H = Integer.parseInt(stk.nextToken());
			stk = new StringTokenizer(br.readLine());
			for (int i = 0; i < n; i++) {
				h[i] = Integer.parseInt(stk.nextToken());
			}
			stk = new StringTokenizer(br.readLine());
			for (int i = 0; i < n-1; i++) {
				d[i] = Integer.parseInt(stk.nextToken());
			}

			/////////////////////////////////////////////////////////////////////////////////////////////
			/*
			   이 부분에서 여러분의 알고리즘이 수행됩니다.
			   문제의 답을 계산하여 그 값을 Answer에 저장하는 것을 가정하였습니다.


			 * 시간 복잡도 분석
			   answers[][]는 2차원 배열로, 생성 후 초기화 단계에서 최대 H번 반복되는 for문이 이용되어 \Theta(H)의 시간이 소요된다.
			   이후 각각 n-1번, H번 반복되는 두 개의 for문 안에서 if문을 통한 비교 연산이 최대 2번 수행되고 이를 통해 나머지 원소를 채운다.
			   이 과정의 전체적인 시간복잡도는 (n-1) \times H \times 2 = \Theta(nH)이다.
			   따라서 전체적인 시간복잡도는 \Theta(nH)이다.
			 */
			/////////////////////////////////////////////////////////////////////////////////////////////

			int[][] answers = new int[H + 1][n]; // answers[hm][i] : 높이가 hm 이하, 최상위 블럭이 i 이하인 경우의 수

			for (int hm = h[0]; hm <= H; hm++) {
				answers[hm][0] = 1;
			}

			for (int i = 1; i < n; i++) {
				for (int hm = 0; hm <= H; hm++) {
					answers[hm][i] = answers[hm][i - 1];
					if (hm >= h[i]) {
						if (i == 1) {
							answers[hm][i] += 1 + answers[hm - h[i] + d[i - 1]][i - 1];
						} else {
							answers[hm][i] += 1 + answers[hm - h[i] + d[i - 1]][i - 1] - answers[hm - h[i] + d[i - 1]][i - 2] + answers[hm - h[i]][i - 2] + 1000000;
						}
					}
					answers[hm][i] %= 1000000;
				}
			}

			Answer = answers[H][n - 1];

			// output4.txt로 답안을 출력합니다.
			pw.println("#" + test_case + " " + Answer);
			/*
			   아래 코드를 수행하지 않으면 여러분의 프로그램이 제한 시간 초과로 강제 종료 되었을 때,
			   출력한 내용이 실제로 파일에 기록되지 않을 수 있습니다.
			   따라서 안전을 위해 반드시 flush() 를 수행하시기 바랍니다.
			 */
			pw.flush();
		}

		br.close();
		pw.close();
	}
}

