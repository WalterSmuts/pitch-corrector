use crate::signal_processing::DisplayProcessor;
use crossterm::event;
use crossterm::event::DisableMouseCapture;
use crossterm::event::EnableMouseCapture;
use crossterm::event::Event;
use crossterm::event::KeyCode;
use crossterm::execute;
use crossterm::terminal;
use crossterm::terminal::disable_raw_mode;
use crossterm::terminal::EnterAlternateScreen;
use crossterm::terminal::LeaveAlternateScreen;
use easyfft::dyn_size::DynFft;
use log::info;
use std::io::Stdout;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;
use tui::backend::CrosstermBackend;
use tui::layout;
use tui::layout::Constraint;
use tui::layout::Direction;
use tui::layout::Layout;
use tui::symbols;
use tui::text::Span;
use tui::widgets::Axis;
use tui::widgets::Block;
use tui::widgets::Borders;
use tui::widgets::Chart;
use tui::widgets::Dataset;
use tui::widgets::GraphType;
use tui::Frame;
use tui::Terminal;
use tui_logger::TuiLoggerSmartWidget;

const BUFFER_SIZE: usize = 1024;

#[derive(PartialEq)]
enum State {
    Display,
    Paused,
    Logger,
}

pub struct UserInterface {
    state: State,
    display_buffers: Vec<Arc<Mutex<[f32; BUFFER_SIZE]>>>,
    frame_rate: Duration,
}

impl UserInterface {
    pub fn run(&mut self) {
        info!("Running UserInterface");
        terminal::enable_raw_mode().unwrap();
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture).unwrap();
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut last_frame = Instant::now();
        loop {
            match self.state {
                State::Display => {
                    terminal.draw(|f| self.draw_frame(f)).unwrap();
                }
                State::Logger => {
                    terminal.draw(|f| self.draw_logger_frame(f)).unwrap();
                }
                State::Paused => (),
            };

            let timeout = self
                .frame_rate
                .checked_sub(last_frame.elapsed())
                .unwrap_or_else(|| Duration::from_secs(0));
            if crossterm::event::poll(timeout).unwrap() {
                if let Event::Key(key) = event::read().unwrap() {
                    if let KeyCode::Char('q') = key.code {
                        return;
                    }
                    if let KeyCode::Char(' ') = key.code {
                        self.state = match self.state {
                            State::Display => State::Paused,
                            _ => State::Display,
                        }
                    }
                    if let KeyCode::Char('l') = key.code {
                        match self.state {
                            State::Display => self.state = State::Logger,
                            _ => self.state = State::Display,
                        }
                    }
                }
            }
            if last_frame.elapsed() >= self.frame_rate {
                last_frame = Instant::now();
            }
        }
    }

    fn draw_logger_frame(&self, frame: &mut Frame<CrosstermBackend<Stdout>>) {
        frame.render_widget(TuiLoggerSmartWidget::default(), frame.size());
    }

    fn draw_frame(&self, frame: &mut Frame<CrosstermBackend<Stdout>>) {
        let mut constraints = Vec::new();
        for _ in 0..self.display_buffers.len() {
            constraints.push(Constraint::Ratio(1, self.display_buffers.len() as u32));
        }
        let outer_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(frame.size());

        for (index, buffer) in self.display_buffers.iter().enumerate() {
            let layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Ratio(1, 2), Constraint::Ratio(1, 2)].as_ref())
                .split(outer_layout[index]);

            let buffer = buffer.lock().unwrap();

            let signal_data: Vec<_> = buffer
                .iter()
                .enumerate()
                .map(|(index, value)| (index as f64, *value as f64))
                .collect();

            let dataset = Dataset::default()
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .data(&signal_data);

            let chart = Chart::new(vec![dataset])
                .block(
                    Block::default()
                        .title("Waveform")
                        .borders(Borders::ALL)
                        .title_alignment(layout::Alignment::Center),
                )
                .x_axis(
                    Axis::default()
                        .title("Time (Samples @ 44100 Hz)")
                        .bounds([0.0, BUFFER_SIZE as f64]),
                )
                .y_axis(
                    Axis::default()
                        .labels(vec![Span::raw("-1.0"), Span::raw("0.0"), Span::raw("1.0")])
                        .title("Amplitude")
                        .bounds([-1.0, 1.0]),
                );

            frame.render_widget(chart, layout[0]);
            let spectrum = buffer.fft();

            let spectrum_data: Vec<_> = spectrum
                .into_iter()
                .enumerate()
                .map(|(index, bin)| (index as f64, bin.norm() as f64))
                .collect();

            let dataset = Dataset::default()
                .marker(symbols::Marker::Braille)
                .graph_type(GraphType::Line)
                .data(&spectrum_data);

            let chart = Chart::new(vec![dataset])
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title_alignment(layout::Alignment::Center)
                        .title("Power Spectral Density"),
                )
                .x_axis(
                    Axis::default()
                        .title("Frequency")
                        .bounds([0.0, BUFFER_SIZE as f64 / 2.0 + 1.0]),
                )
                .y_axis(
                    Axis::default()
                        .labels(vec![Span::raw("0.0"), Span::raw("25.0"), Span::raw("50.0")])
                        .title("Power")
                        .bounds([0.0, 50.0]),
                );

            frame.render_widget(chart, layout[1]);
        }
    }

    pub fn create_display_processor(&mut self) -> DisplayProcessor {
        let display_processor = DisplayProcessor::new();
        self.display_buffers
            .push(display_processor.clone_display_buffer());
        display_processor
    }
}

impl UserInterface {
    pub fn new() -> Self {
        info!("Creating new UserInterface");
        Self {
            state: State::Display,
            frame_rate: Duration::from_millis(10),
            display_buffers: Vec::new(),
        }
    }
}

impl Drop for UserInterface {
    fn drop(&mut self) {
        info!("Dropping user UserInterface");
        let stdout = std::io::stdout();
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend).unwrap();
        disable_raw_mode().unwrap();
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )
        .unwrap();
        terminal.show_cursor().unwrap();
    }
}
