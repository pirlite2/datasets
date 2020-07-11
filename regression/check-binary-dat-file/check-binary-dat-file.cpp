// Test utility to print binary DAT file -- pir -- 10.7.2020

// REVISION HISTORY:

//*****************************************************************************
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
//  MA 02110-1301, USA.
//
//*******************************************************************************

#include <iostream>

//#include <CDataset/CDataset.h>
#include "CDataset-pir.h"

using namespace std;

//*****************************************************************************

int main(int argc, char* argv[])
	{
	if(argc != 2)
		{
		cout << "Incorrect number of command line arguments supplied" << endl;
		exit(1);
		}
	cout << "Printing "	<< argv[1] << endl;

	CDataset<double> dataset;
	dataset.Load(argv[1]);








	return 0;
	} // main()

//*****************************************************************************
