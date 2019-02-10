INCDIR=-I$(HOME)/.local/include/
LIBDIR=-L$(HOME)/.local/lib/
LIBS=-lgsl -lgslcblas -lconfig++ -lergopack -lgomp -lgslcblas
CONFIG=../cfg/readConfig
MODEL=../cfg/coupledRO

CC=g++
WARN=-Wall -Wformat=0
OMPFLAGS=-fopenmp
CFLAGS=$(WARN) -pedantic-errors -O3
LDFLAGSSTD=-undefined dynamic_lookup
LDFLAGSGDB=$(LDFLAGSSTD) -gdb

LDFLAGS=$(LDFLAGSSTD)
#LDFLAGS=$(LDFLAGSGDB)

all:$(EXE).out
	mkdir -p ../results/simulation

$(EXE).o:$(EXE).cpp
	$(CC) $(OMPFLAGS) -c $(CFLAGS) $(INCDIR) -o $@ $<

$(CONFIG).o:$(CONFIG).cpp
	$(CC) $(OMPFLAGS) -c $(CFLAGS) $(INCDIR) -o $@ $<

$(MODEL).o:$(MODEL).cpp
	$(CC) $(OMPFLAGS) -c $(CFLAGS) $(INCDIR) -o $@ $<

$(EXE).out:$(EXE).o $(CONFIG).o $(MODEL).o
	$(CC) $(OMPFLAGS) $(LDFLAGS) -o $@ $^ $(LIBDIR) $(LIBS)

clean:
	rm -f $(EXE).o $(EXE).out

cleanall:
	rm -f $(EXE).o $(CONFIG).o $(MODEL).o $(EXE).out
