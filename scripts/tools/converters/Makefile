CPP_FILES := $(wildcard *.cpp)
EXES := $(CPP_FILES:.cpp=)

CXX := g++
CXXFLAGS := -O2

all: $(EXES)

%: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(EXES)