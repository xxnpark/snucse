import java.io.*;
import java.util.*;

public class SortingTest
{
	public static void main(String args[])
	{
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

		try
		{
			boolean isRandom = false;	// 입력받은 배열이 난수인가 아닌가?
			int[] value;	// 입력 받을 숫자들의 배열
			String nums = br.readLine();	// 첫 줄을 입력 받음
			if (nums.charAt(0) == 'r')
			{
				// 난수일 경우
				isRandom = true;	// 난수임을 표시

				String[] nums_arg = nums.split(" ");

				int numsize = Integer.parseInt(nums_arg[1]);	// 총 갯수
				int rminimum = Integer.parseInt(nums_arg[2]);	// 최소값
				int rmaximum = Integer.parseInt(nums_arg[3]);	// 최대값

				Random rand = new Random();	// 난수 인스턴스를 생성한다.

				value = new int[numsize];	// 배열을 생성한다.
				for (int i = 0; i < value.length; i++)	// 각각의 배열에 난수를 생성하여 대입
					value[i] = rand.nextInt(rmaximum - rminimum + 1) + rminimum;
			}
			else
			{
				// 난수가 아닐 경우
				int numsize = Integer.parseInt(nums);

				value = new int[numsize];	// 배열을 생성한다.
				for (int i = 0; i < value.length; i++)	// 한줄씩 입력받아 배열원소로 대입
					value[i] = Integer.parseInt(br.readLine());
			}

			// 숫자 입력을 다 받았으므로 정렬 방법을 받아 그에 맞는 정렬을 수행한다.
			while (true)
			{
				int[] newvalue = (int[])value.clone();	// 원래 값의 보호를 위해 복사본을 생성한다.

				String command = br.readLine();

				long t = System.currentTimeMillis();
				switch (command.charAt(0))
				{
					case 'B':	// Bubble Sort
						DoBubbleSort(newvalue);
						break;
					case 'I':	// Insertion Sort
						DoInsertionSort(newvalue);
						break;
					case 'H':	// Heap Sort
						newvalue = DoHeapSort(newvalue);
						break;
					case 'M':	// Merge Sort
						DoMergeSort(newvalue, 0, newvalue.length - 1);
						break;
					case 'Q':	// Quick Sort
						DoQuickSort(newvalue, 0, newvalue.length - 1);
						break;
					case 'R':	// Radix Sort
						newvalue = DoRadixSort(newvalue);
						break;
					case 'X':
						return;	// 프로그램을 종료한다.
					default:
						throw new IOException("잘못된 정렬 방법을 입력했습니다.");
				}
				if (isRandom)
				{
					// 난수일 경우 수행시간을 출력한다.
					System.out.println((System.currentTimeMillis() - t) + " ms");
				}
				else
				{
					/*
					for (int i = 0; i < newvalue.length; i++)
					{
						System.out.println(newvalue[i]);
					}
					*/

					// 난수가 아닐 경우 정렬된 결과값을 출력한다.
					StringBuilder printStr = new StringBuilder();
					for (int num : newvalue) {
						printStr.append(num).append("\n");
					}
					System.out.println(printStr.deleteCharAt(printStr.length() - 1));
				}

			}
		}
		catch (IOException e)
		{
			System.out.println("입력이 잘못되었습니다. 오류 : " + e.toString());
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	private static void swap(int[] arr, int i, int j) {
		int temp = arr[i];
		arr[i] = arr[j];
		arr[j] = temp;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	private static void DoBubbleSort(int[] value) {
		// value는 정렬안된 숫자들의 배열이며 value.length 는 배열의 크기가 된다.
		// 결과로 정렬된 배열은 리턴해 주어야 하며, 두가지 방법이 있으므로 잘 생각해서 사용할것.
		// 주어진 value 배열에서 안의 값만을 바꾸고 value를 다시 리턴하거나
		// 같은 크기의 새로운 배열을 만들어 그 배열을 리턴할 수도 있다.

		for (int l = value.length - 1; l > 0; l--) {
			for (int i = 0; i < l; i++) {
				if (value[i] > value[i + 1]) {
					swap(value, i, i + 1);
				}
			}
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	private static void DoInsertionSort(int[] value) {
		for (int i = 1; i < value.length; i++) {
			int num = value[i];
			for (int j = i - 1; j >= 0 && num < value[j]; j--) {
				swap(value, j, j + 1);
			}
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	private static int[] DoHeapSort(int[] value) {
		buildHeap(value);

		for (int i = value.length - 1; i > 0; i--) {
			value[i] = deleteMax(value, i + 1);
		}

		return (value);
	}

	private static void percolateDown(int[] arr, int k, int n) {
		int child = 2 * k + 1;
		int right = 2 * k + 2;
		if (child < n) {
			if (right < n && arr[child] < arr[right]) {
				child = right;
			}
			if (arr[k] < arr[child]) {
				swap(arr, k, child);
				percolateDown(arr, child, n);
			}
		}
	}

	private static int deleteMax(int[] arr, int n) {
		int max = arr[0];
		arr[0] = arr[n - 1];
		percolateDown(arr, 0, n - 1);
		return max;
	}

	private static void buildHeap(int[] arr) {
		for (int i = arr.length / 2 - 1; i >= 0; i--) {
			percolateDown(arr, i, arr.length);
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	private static void DoMergeSort(int[] value, int p, int r) {
		if (p < r) {
			int q = (p + r) / 2;
			DoMergeSort(value, p, q);
			DoMergeSort(value, q + 1, r);
			merge(value, p, q, r);
		}
	}

	private static void merge(int[] arr, int p, int q, int r) {
		if (arr[q] < arr[q + 1]) {
			return;
		}

		int i = p; int j = q + 1; int t = 0;
		int[] newArr = new int[r - p + 1];

		while (i <= q && j <= r) {
			if (arr[i] <= arr[j]) {
				newArr[t++] = arr[i++];
			} else {
				newArr[t++] = arr[j++];
			}
		}

		while (i <= q) {
			newArr[t++] = arr[i++];
		}

		while (j <= r) {
			newArr[t++] = arr[j++];
		}

		for (int k = 0; k <= r - p; k++) {
			arr[p + k] = newArr[k];
		}
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	private static void DoQuickSort(int[] value, int p, int r) {
//		System.out.println(Arrays.toString(value));
		if (p < r) {
			int q = partition(value, p, r);
			DoQuickSort(value, p, q - 1);
			DoQuickSort(value, q + 1, r);
		}
	}

	private static int partition(int[] arr, int p, int r) {
		int x = arr[r];
		int i = p - 1;

		for (int j = p; j < r; j++) {
			if (arr[j] < x) {
				swap(arr, ++i, j);
			}
		}
		swap(arr, i + 1, r);

		return i + 1;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////
	private static int[] DoRadixSort(int[] value) {
		int positiveLen = 0;
		int negativeLen = 0;
		int[] positiveValue = new int[value.length];
		int[] negativeValue = new int[value.length];
		Arrays.fill(positiveValue, 0);
		Arrays.fill(negativeValue, 0);

		for (int num : value) {
			if (num >= 0) {
				positiveValue[positiveLen++] = num;
			} else {
				negativeValue[negativeLen++] = -num;
			}
		}

		int positiveMax = maxIndex(positiveValue);
		for (int i = 0; i < positiveMax; i++) {
			positiveValue = indexSort(positiveValue, i);
		}

		int negativeMax = maxIndex(negativeValue);
		for (int i = 0; i < negativeMax; i++) {
			negativeValue = indexSort(negativeValue, i);
		}

		for (int i = 0; i < negativeLen; i++) {
			value[i] = -negativeValue[value.length - i - 1];
		}

		for (int i = 0; i < positiveLen; i++) {
			value[negativeLen + i] = positiveValue[negativeLen + i];
		}

		return (value);
	}

	private static int maxIndex(int[] value) {
		int max = 0;

		for (int n : value) {
			int length = Integer.toString(n).length();
			if (max < length) {
				max = length;
			}
		}

		return max;
	}

	private static int[] indexSort(int[] value, int index) {
		int[] count = new int[10]; // 각 자리수별 개수
		int[] ret = new int[value.length];

		for (int num : value) {
			int n = (num / (int) Math.pow(10, index)) % 10;
			count[n]++;
		}

		for (int i = 0; i < 9; i++) {
			count[i + 1] += count[i];
		}

		for (int i = value.length - 1; i >= 0; i--) {
			ret[count[(value[i] / (int) Math.pow(10, index)) % 10]-- - 1] = value[i];
		}

		return ret;
	}
}
