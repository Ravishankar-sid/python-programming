import Foundation

struct Printer<T> {
	/*
		* Name: printArray
		* Print each element of the generic array on a new line. Do not return anything.
		* @param A generic array
	*/

	func printArray<Int>(array: [Int]) {
		for a in array{
			print(a)
		}
    }

}
