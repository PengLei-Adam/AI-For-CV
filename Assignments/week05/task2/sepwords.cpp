/*
Separate string into vector of words
Suppose a word is only consist of "a-zA-z"
*/

#include <string>
#include <vector>
#include <iostream>

using std::string;
using std::vector;
using std::cout;
using std::endl;

bool iswordchar(char c)
{
	return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z');
		
}
void sepwords(string & sentence, vector<string> & words)
{
	if (!words.empty())
		words.clear();
	size_t insert_id = 0;
	for (string::iterator it = sentence.begin(); it != sentence.end(); ++it)
	{

		if (iswordchar(*it))
		{
			if (words.size() == insert_id) {
				words.push_back(string());
			}
			words[insert_id].push_back(*it);
		}
		else if (insert_id == words.size() - 1)
		{
			++insert_id;
		}
	}
}

int main()
{
	string sent("I am happy and  good.he is now");
	vector<string> words;
	sepwords(sent, words);
	for (vector<string>::iterator wi = words.begin(); wi != words.end(); ++wi)
	{
		cout << *wi << endl;
	}
	return 0;
}
